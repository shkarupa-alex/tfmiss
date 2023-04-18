import tensorflow as tf
from keras.optimizers import Optimizer
from keras.saving import register_keras_serializable
from keras.src.optimizers.optimizer import base_optimizer_keyword_args


@register_keras_serializable(package='Miss')
class Lion(Optimizer):
    """
    Inspired with https://github.com/google/automl/blob/master/lion/lion_tf2.py

    Proposed in "Symbolic Discovery of Optimization Algorithms"
    https://arxiv.org/abs/2302.06675

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
          `keras.optimizers.schedules.LearningRateSchedule`, or a callable that takes no arguments and returns the
          actual value to use. The learning rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns the
          actual value to use. The exponential decay rate for the 1st moment estimates. Defaults to 0.9.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns the
          actual value to use. The exponential decay rate for the 2nd moment estimates. Defaults to 0.99.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
            self, learning_rate=0.0001, beta_1=0.9, beta_2=0.99, weight_decay=None, clipnorm=None, clipvalue=None,
            global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=True,
            name='Lion', **kwargs):
        super().__init__(
            name=name, weight_decay=weight_decay, clipnorm=clipnorm, clipvalue=clipvalue,
            global_clipnorm=global_clipnorm, use_ema=use_ema, ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency, jit_compile=jit_compile, **kwargs)

        self._learning_rate = self._build_learning_rate(learning_rate)
        self._beta_1 = beta_1
        self._beta_2 = beta_2

    def build(self, var_list):
        super().build(var_list)
        if getattr(self, '_built', False):
            return

        self.moments = []

        for var in var_list:
            self.moments.append(self.add_variable_from_reference(var, 'm'))

        self._built = True

    def update_step(self, gradient, variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1 = tf.cast(self._beta_1, variable.dtype)
        beta_2 = tf.cast(self._beta_2, variable.dtype)

        inv_beta_1 = 1. - beta_1
        inv_beta_2 = 1. - beta_2

        var_key = self._var_key(variable)
        var_idx = self._index_dict[var_key]

        m = self.moments[var_idx]

        if isinstance(gradient, tf.IndexedSlices):
            sparse_m_t = tf.gather(m, gradient.indices)

            var_update = lr * tf.math.sign(sparse_m_t * beta_1 + gradient.values * inv_beta_1)
            variable.scatter_sub(tf.IndexedSlices(var_update, gradient.indices))

            m_update = sparse_m_t * beta_2 + gradient.values * inv_beta_2
            m.scatter_update(tf.IndexedSlices(m_update, gradient.indices))
        else:
            var_update = lr * tf.math.sign(m * beta_1 + gradient * inv_beta_1)
            variable.assign_sub(var_update)

            m_update = m * beta_2 + gradient * inv_beta_2
            m.assign(m_update)

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter(self._learning_rate),
            'beta_1': self._beta_1,
            'beta_2': self._beta_2
        })

        return config


Lion.__doc__ = Lion.__doc__.replace("{{base_optimizer_keyword_args}}", base_optimizer_keyword_args)
