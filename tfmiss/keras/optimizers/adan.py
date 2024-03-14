import tensorflow as tf
from keras.optimizers import Optimizer
from keras.saving import register_keras_serializable
from keras.src.backend.common import KerasVariable
from keras.src.optimizers.optimizer import base_optimizer_keyword_args


@register_keras_serializable(package='Miss')
class Adan(Optimizer):
    """
    Inspired with https://github.com/DenisVorotyntsev/Adan/blob/main/tf_adan/adan.py

    Proposed in "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
    https://arxiv.org/abs/2208.06677

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
          `keras.optimizers.schedules.LearningRateSchedule`, or a callable that takes no arguments and returns the
          actual value to use. The learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns the
          actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to 0.98.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns the
          actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to 0.92.
        beta_3: A float value or a constant float tensor, or a callable that takes no arguments and returns the
          actual value to use. The exponential decay rate for the 3rd moment estimates. Defaults to 0.99.
        epsilon: A small constant for numerical stability. Defaults to 1e-8.
        sparse_support: A boolean flag, support or not sparse updates. Setting to False can reduce memory
          consumption up to 25%.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
            self, learning_rate=0.001, beta_1=0.98, beta_2=0.92, beta_3=0.99, epsilon=1e-8, sparse_support=False,
            name='Adan', weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
            ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None,
            **kwargs):
        super().__init__(
            learning_rate, name=name, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue,
            global_clipnorm=global_clipnorm, use_ema=use_ema, ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency, loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps, **kwargs)

        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._beta_3 = beta_3
        self._epsilon = epsilon
        self._sparse_support = sparse_support

    def build(self, var_list):
        super().build(var_list)
        if getattr(self, '_built', False):
            return

        self.exp_avgs = []
        self.exp_avg_diffs = []
        self.exp_avg_sqs = []
        self.prev_grads = []
        self.sparse_steps = []

        for var in var_list:
            self.exp_avgs.append(self.add_variable_from_reference(var, 'exp_avg'))
            self.exp_avg_diffs.append(self.add_variable_from_reference(var, 'exp_avg_diff'))
            self.exp_avg_sqs.append(self.add_variable_from_reference(var, 'exp_avg_sq'))
            self.prev_grads.append(self.add_variable_from_reference(var, 'prev_grad'))

            if self._sparse_support and len(var.shape):
                if hasattr(var, 'path'):
                    name = var.path.replace('/', '_') + '_var'
                else:
                    name = str(var.name).replace(':', '_') + '_var'
                self.sparse_steps.append(
                    self.add_variable(shape=var.shape[:1], initializer='zeros', dtype=var.dtype, name=name))

    def update_step(self, gradient, variable, learning_rate):
        if isinstance(gradient, tf.IndexedSlices) and not self._sparse_support:
            raise ValueError('Optimizer not configured to support sparse updates.')

        lr = tf.cast(learning_rate, variable.dtype)
        beta_1 = tf.cast(self._beta_1, variable.dtype)
        beta_2 = tf.cast(self._beta_2, variable.dtype)
        beta_3 = tf.cast(self._beta_3, variable.dtype)
        epsilon_2 = tf.cast(self._epsilon ** 2, variable.dtype)

        var_idx = self._get_variable_index(variable)

        exp_avg = self.exp_avgs[var_idx]
        exp_avg_sq = self.exp_avg_sqs[var_idx]
        exp_avg_diff = self.exp_avg_diffs[var_idx]
        prev_grad = self.prev_grads[var_idx]
        sparse_step = self.sparse_steps[var_idx] if self._sparse_support else None

        if isinstance(gradient, tf.IndexedSlices):
            sparse_step_t = tf.gather(sparse_step, gradient.indices)
            sparse_step_t = tf.reshape(sparse_step_t, [-1] + [1] * (variable.shape.rank - 1))

            update_step_t = sparse_step_t + 1.
            prev_grad_t = tf.where(0. == sparse_step_t, gradient.values, tf.gather(prev_grad, gradient.indices))
            diff_t = gradient.values - prev_grad_t
            update_t = gradient.values + beta_2 * diff_t
            exp_avg_t = tf.gather(exp_avg, gradient.indices) * beta_1 + gradient.values * (1. - beta_1)
            exp_avg_diff_t = tf.gather(exp_avg_diff, gradient.indices) * beta_2 + diff_t * (1. - beta_2)
            exp_avg_sq_t = tf.gather(exp_avg_sq, gradient.indices) * beta_3 + update_t ** 2 * (1. - beta_3)
        else:
            update_step_t = tf.cast(self.iterations + 1, variable.dtype)
            prev_grad_t = tf.cond(
                self.iterations == 0, lambda: tf.identity(gradient), lambda: tf.identity(prev_grad))
            diff_t = gradient - prev_grad_t
            update_t = gradient + beta_2 * diff_t
            exp_avg_t = exp_avg * beta_1 + gradient * (1. - beta_1)
            exp_avg_diff_t = exp_avg_diff * beta_2 + diff_t * (1. - beta_2)
            exp_avg_sq_t = exp_avg_sq * beta_3 + update_t ** 2 * (1. - beta_3)

        bias_correction_1 = 1. / (1. - tf.pow(beta_1, update_step_t))
        bias_correction_2 = 1. / (1. - tf.pow(beta_2, update_step_t))
        bias_correction_3 = 1. / (1. - tf.pow(beta_3, update_step_t))
        var_t = (exp_avg_t * bias_correction_1 + beta_2 * exp_avg_diff_t * bias_correction_2)
        var_t *= tf.math.rsqrt(exp_avg_sq_t * bias_correction_3 + epsilon_2)
        var_t = - var_t * lr

        if isinstance(gradient, tf.IndexedSlices):

            self._scatter_update(exp_avg, tf.IndexedSlices(exp_avg_t, gradient.indices))
            self._scatter_update(exp_avg_diff, tf.IndexedSlices(exp_avg_diff_t, gradient.indices))
            self._scatter_update(exp_avg_sq, tf.IndexedSlices(exp_avg_sq_t, gradient.indices))
            self._scatter_update(prev_grad, gradient)
            self.assign_add(variable, tf.IndexedSlices(var_t, gradient.indices))
            self.assign_add(sparse_step, tf.IndexedSlices(
                tf.ones_like(gradient.indices, dtype=variable.dtype), gradient.indices))
        else:

            self.assign(exp_avg, exp_avg_t)
            self.assign(exp_avg_diff, exp_avg_diff_t)
            self.assign(exp_avg_sq, exp_avg_sq_t)
            self.assign(prev_grad, gradient)
            self.assign_add(variable, var_t)

    def _scatter_update(self, variable, value):
        if isinstance(variable, KerasVariable):
            variable = variable.value
        value = tf.cast(value, variable.dtype)

        if not isinstance(value, tf.IndexedSlices):
            raise ValueError(f'Expected value of `tf.IndexedSlices`, got {type(value)}')

        variable.scatter_update(value)

    def get_config(self):
        config = super().get_config()
        config.update({
            'beta_1': self._beta_1,
            'beta_2': self._beta_2,
            'beta_3': self._beta_3,
            'epsilon': self._epsilon,
            'sparse_support': self._sparse_support
        })

        return config


Adan.__doc__ = Adan.__doc__.replace("{{base_optimizer_keyword_args}}", base_optimizer_keyword_args)
