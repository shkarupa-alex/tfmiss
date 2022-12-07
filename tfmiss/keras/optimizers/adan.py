import tensorflow as tf
from keras import backend
from keras.optimizers.optimizer_v2 import optimizer_v2
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='Miss')
class Adan(optimizer_v2.OptimizerV2):
    def __init__(
            self, learning_rate=0.001, weight_decay=0.0, beta_1=0.98, beta_2=0.92, beta_3=0.99, epsilon=1e-8,
            sparse_support=True, name='Adan', **kwargs):
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
            weight_decay: A `tf.Tensor`, floating point value. The weight decay. Defaults to 0.0.
            sparse_support: A boolean flag, support or not sparse updates. Setting to False can reduce memory
              consumption up to 25%.
            name (str, optional): optimizer name. Defaults to "Adan".
        """
        super().__init__(name=name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('beta_3', beta_3)
        self.epsilon = epsilon or backend.epsilon()
        self.sparse_support = sparse_support

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'exp_avg')
            self.add_slot(var, 'exp_avg_diff')
            self.add_slot(var, 'exp_avg_sq')
            self.add_slot(var, 'prev_grad')

            if self.sparse_support and len(var.shape):
                self.add_slot(var, 'sparse_step', initializer='ones', shape=var.shape[:1] + (1,))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        first_step = self.iterations == 0
        dense_step = tf.cast(self.iterations + 1, var_dtype)
        weight_decay = tf.identity(self._get_hyper('weight_decay', var_dtype))
        decay_norm = 1. / (1. + apply_state[(var_device, var_dtype)]['lr_t'] * weight_decay)
        beta_1 = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2 = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_3 = tf.identity(self._get_hyper('beta_3', var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                first_step=first_step,
                dense_step=dense_step,
                decay_norm=decay_norm,
                beta_1=beta_1,
                beta_2=beta_2,
                beta_3=beta_3,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or \
                       self._fallback_apply_state(var_device, var_dtype)

        exp_avg = self.get_slot(var, 'exp_avg')
        exp_avg_sq = self.get_slot(var, 'exp_avg_sq')
        exp_avg_diff = self.get_slot(var, 'exp_avg_diff')
        prev_grad = self.get_slot(var, 'prev_grad')

        first_step = coefficients['first_step']
        dense_step = coefficients['dense_step']
        lr = coefficients['lr_t']
        decay_norm = coefficients['decay_norm']
        beta_1 = coefficients['beta_1']
        beta_2 = coefficients['beta_2']
        beta_3 = coefficients['beta_3']
        epsilon = coefficients['epsilon']

        prev_grad_t = smart_cond(first_step, lambda: tf.identity(grad), lambda: tf.identity(prev_grad))
        diff = grad - prev_grad_t
        update = grad + beta_2 * diff

        exp_avg_t = exp_avg * beta_1 + grad * (1. - beta_1)
        exp_avg_diff_t = exp_avg_diff * beta_2 + diff * (1. - beta_2)
        exp_avg_sq_t = exp_avg_sq * beta_3 + update ** 2 * (1. - beta_3)

        bias_correction_1 = 1.0 - tf.pow(beta_1, dense_step)
        bias_correction_2 = 1.0 - tf.pow(beta_2, dense_step)
        bias_correction_3 = 1.0 - tf.pow(beta_3, dense_step)

        var_t = (exp_avg_t / bias_correction_1 + beta_2 * exp_avg_diff_t / bias_correction_2)
        var_t /= tf.math.sqrt(exp_avg_sq_t / bias_correction_3) + epsilon
        var_t = (var - var_t * lr) * decay_norm

        exp_avg_update = exp_avg.assign(exp_avg_t, use_locking=self._use_locking, read_value=False)
        exp_avg_diff_update = exp_avg_diff.assign(exp_avg_diff_t, use_locking=self._use_locking, read_value=False)
        exp_avg_sq_update = exp_avg_sq.assign(exp_avg_sq_t, use_locking=self._use_locking, read_value=False)
        prev_grad_update = prev_grad.assign(grad, use_locking=self._use_locking, read_value=False)
        var_update = var.assign(var_t, use_locking=self._use_locking, read_value=False)

        return tf.group(exp_avg_update, exp_avg_diff_update, exp_avg_sq_update, prev_grad_update, var_update)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if not self.sparse_support:
            raise ValueError('Optimizer not configured to support sparse updates.')

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or \
                       self._fallback_apply_state(var_device, var_dtype)

        exp_avg = self.get_slot(var, 'exp_avg')
        exp_avg_sq = self.get_slot(var, 'exp_avg_sq')
        exp_avg_diff = self.get_slot(var, 'exp_avg_diff')
        prev_grad = self.get_slot(var, 'prev_grad')
        sparse_step = self.get_slot(var, 'sparse_step')

        lr = coefficients['lr_t']
        decay_norm = coefficients['decay_norm']
        beta_1 = coefficients['beta_1']
        beta_2 = coefficients['beta_2']
        beta_3 = coefficients['beta_3']
        epsilon = coefficients['epsilon']

        sparse_step_t = tf.gather(sparse_step, indices)
        prev_grad_t = tf.gather(prev_grad, indices)
        prev_grad_t = tf.where(1. == sparse_step_t, tf.identity(grad), tf.identity(prev_grad_t))

        diff = grad - prev_grad_t
        update = grad + beta_2 * diff

        exp_avg_t = tf.gather(exp_avg, indices) * beta_1 + grad * (1. - beta_1)
        exp_avg_diff_t = tf.gather(exp_avg_diff, indices) * beta_2 + diff * (1. - beta_2)
        exp_avg_sq_t = tf.gather(exp_avg_sq, indices) * beta_3 + update ** 2 * (1. - beta_3)

        bias_correction_1 = 1.0 - tf.pow(beta_1, sparse_step_t)
        bias_correction_2 = 1.0 - tf.pow(beta_2, sparse_step_t)
        bias_correction_3 = 1.0 - tf.pow(beta_3, sparse_step_t)

        var_t = (exp_avg_t / bias_correction_1 + beta_2 * exp_avg_diff_t / bias_correction_2)
        var_t /= tf.math.sqrt(exp_avg_sq_t / bias_correction_3) + epsilon
        var_t = (tf.gather(var, indices) - var_t * lr) * decay_norm

        exp_avg_update = self._resource_scatter_update(exp_avg, indices, exp_avg_t)
        exp_avg_diff_update = self._resource_scatter_update(exp_avg_diff, indices, exp_avg_diff_t)
        exp_avg_sq_update = self._resource_scatter_update(exp_avg_sq, indices, exp_avg_sq_t)
        prev_grad_update = self._resource_scatter_update(prev_grad, indices, grad)
        var_update = self._resource_scatter_update(var, indices, var_t)
        sparse_step_update = self._resource_scatter_add(
            sparse_step, indices, tf.ones_like(sparse_step_t, dtype=var.dtype))

        return tf.group(
            exp_avg_update, exp_avg_diff_update, exp_avg_sq_update, prev_grad_update, var_update, sparse_step_update)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'weight_decay': self._serialize_hyperparameter('weight_decay'),
                'beta_1': self._serialize_hyperparameter('beta_1'),
                'beta_2': self._serialize_hyperparameter('beta_2'),
                'beta_3': self._serialize_hyperparameter('beta_3'),
                'epsilon': self.epsilon,
                'sparse_support': self.sparse_support
            }
        )
        return config
