import tensorflow as tf
from keras.optimizers.optimizer_experimental.optimizer import Optimizer, base_optimizer_keyword_args
from keras.utils.control_flow_util import smart_cond
from keras.saving.object_registration import register_keras_serializable


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
            self, learning_rate=0.001, beta_1=0.98, beta_2=0.92, beta_3=0.99, epsilon=1e-8, sparse_support=True,
            weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99,
            ema_overwrite_frequency=None, jit_compile=True, name='Adan', **kwargs):
        super().__init__(
            name=name, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue,
            global_clipnorm=global_clipnorm, use_ema=use_ema, ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency, jit_compile=jit_compile, **kwargs)

        self._learning_rate = self._build_learning_rate(learning_rate)
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
                self.sparse_steps.append(self.add_variable_from_reference(var, 'sparse_step', shape=var.shape[:1]))

        self._built = True

    def update_step(self, gradient, variable):
        if isinstance(gradient, tf.IndexedSlices) and not self._sparse_support:
            raise ValueError('Optimizer not configured to support sparse updates.')

        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1 = tf.cast(self._beta_1, variable.dtype)
        beta_2 = tf.cast(self._beta_2, variable.dtype)
        beta_3 = tf.cast(self._beta_3, variable.dtype)
        epsilon_2 = tf.cast(self._epsilon ** 2, variable.dtype)

        var_key = self._var_key(variable)
        var_idx = self._index_dict[var_key]

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
            prev_grad_t = smart_cond(
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
            exp_avg.scatter_update(tf.IndexedSlices(exp_avg_t, gradient.indices))
            exp_avg_diff.scatter_update(tf.IndexedSlices(exp_avg_diff_t, gradient.indices))
            exp_avg_sq.scatter_update(tf.IndexedSlices(exp_avg_sq_t, gradient.indices))
            prev_grad.scatter_update(gradient)
            variable.scatter_add(tf.IndexedSlices(var_t, gradient.indices))
            sparse_step.scatter_add(tf.IndexedSlices(
                tf.ones_like(gradient.indices, dtype=variable.dtype), gradient.indices))
        else:
            exp_avg.assign(exp_avg_t)
            exp_avg_diff.assign(exp_avg_diff_t)
            exp_avg_sq.assign(exp_avg_sq_t)
            prev_grad.assign(gradient)
            variable.assign_add(var_t)

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'beta_1': self._beta_1,
            'beta_2': self._beta_2,
            'beta_3': self._beta_3,
            'epsilon': self._epsilon,
            'sparse_support': self._sparse_support
        })

        return config


Adan.__doc__ = Adan.__doc__.replace("{{base_optimizer_keyword_args}}", base_optimizer_keyword_args)
