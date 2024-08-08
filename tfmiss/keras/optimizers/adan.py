import tensorflow as tf
from keras.src import ops
from keras.src.optimizers import Optimizer
from keras.src.optimizers.optimizer import base_optimizer_keyword_args
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Miss")
class Adan(Optimizer):
    """
    Inspired with
    https://github.com/DenisVorotyntsev/Adan/blob/main/tf_adan/adan.py

    Proposed in "Adan: Adaptive Nesterov Momentum Algorithm for Faster
    Optimizing Deep Models"
    https://arxiv.org/abs/2208.06677

    Args:
        learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
          `keras.optimizers.schedules.LearningRateSchedule`, or a callable that
          takes no arguments and returns the actual value to use. The learning
          rate. Defaults to 0.001.
        beta_1: A float value or a constant float tensor, or a callable that
          takes no arguments and returns the actual value to use. The
          exponential decay rate for the 1st moment estimates. Defaults to 0.98.
        beta_2: A float value or a constant float tensor, or a callable that
          takes no arguments and returns the actual value to use.
          The exponential decay rate for the 2nd moment estimates. Defaults
          to 0.92.
        beta_3: A float value or a constant float tensor, or a callable that
          takes no arguments and returns the actual value to use. The
          exponential decay rate for the 3rd moment estimates. Defaults to 0.99.
        epsilon: A small constant for numerical stability. Defaults to 1e-8.
        {{base_optimizer_keyword_args}}
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.98,
        beta_2=0.92,
        beta_3=0.99,
        epsilon=1e-8,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="Adan",
        **kwargs
    ):
        super().__init__(
            learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._beta_3 = beta_3
        self._epsilon = epsilon

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

        self.exp_avgs = []
        self.exp_avg_diffs = []
        self.exp_avg_sqs = []
        self.prev_grads = []

        for var in var_list:
            self.exp_avgs.append(
                self.add_variable_from_reference(var, "exp_avg")
            )
            self.exp_avg_diffs.append(
                self.add_variable_from_reference(var, "exp_avg_diff")
            )
            self.exp_avg_sqs.append(
                self.add_variable_from_reference(var, "exp_avg_sq")
            )
            self.prev_grads.append(
                self.add_variable_from_reference(var, "prev_grad")
            )

    def update_step(self, gradient, variable, learning_rate):
        var_idx = self._get_variable_index(variable)
        exp_avg = self.exp_avgs[var_idx]
        exp_avg_sq = self.exp_avg_sqs[var_idx]
        exp_avg_diff = self.exp_avg_diffs[var_idx]
        prev_grad = self.prev_grads[var_idx]

        local_step = ops.cast(self.iterations + 1, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        prev_grad_t = ops.cond(
            0 == self.iterations, lambda: gradient, lambda: prev_grad
        )

        exp_avg_t, exp_avg_diff_t, exp_avg_sq_t, var_t = self._update_step(
            variable,
            learning_rate,
            local_step,
            gradient,
            exp_avg,
            exp_avg_sq,
            exp_avg_diff,
            prev_grad_t,
        )

        self.assign(exp_avg, exp_avg_t)
        self.assign(exp_avg_diff, exp_avg_diff_t)
        self.assign(exp_avg_sq, exp_avg_sq_t)
        self.assign(prev_grad, gradient)
        self.assign_add(variable, var_t)

    def _update_step(
        self,
        variable,
        learning_rate,
        local_step,
        gradient,
        exp_avg,
        exp_avg_sq,
        exp_avg_diff,
        prev_grad,
    ):
        lr = ops.cast(learning_rate, variable.dtype)
        beta_1 = ops.cast(self._beta_1, variable.dtype)
        beta_2 = ops.cast(self._beta_2, variable.dtype)
        beta_3 = ops.cast(self._beta_3, variable.dtype)
        epsilon_2 = ops.cast(self._epsilon**2, variable.dtype)

        diff_t = ops.subtract(gradient, prev_grad)
        update_t = gradient + diff_t * beta_2
        exp_avg = exp_avg * beta_1 + gradient * (1.0 - beta_1)
        exp_avg_diff = exp_avg_diff * beta_2 + diff_t * (1.0 - beta_2)
        exp_avg_sq = exp_avg_sq * beta_3 + ops.square(update_t) * (1.0 - beta_3)

        bias_correction_1 = 1.0 - ops.power(beta_1, local_step)
        bias_correction_2 = 1.0 - ops.power(beta_2, local_step)
        bias_correction_3 = 1.0 - ops.power(beta_3, local_step)
        var_t = (
            -lr
            * (
                exp_avg / bias_correction_1
                + exp_avg_diff * beta_2 / bias_correction_2
            )
            * ops.rsqrt(exp_avg_sq / bias_correction_3 + epsilon_2)
        )

        return exp_avg, exp_avg_diff, exp_avg_sq, var_t

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta_1": self._beta_1,
                "beta_2": self._beta_2,
                "beta_3": self._beta_3,
                "epsilon": self._epsilon,
            }
        )

        return config


@register_keras_serializable(package="Miss")
class LazyAdan(Adan):
    """
    LazyAdan is a variant of the Adan optimizer that handles sparse updates
    more efficiently. However, it provides slightly different semantics than
    the original Adan algorithm, and may lead to different empirical results.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.98,
        beta_2=0.92,
        beta_3=0.99,
        epsilon=1e-8,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="LazyAdan",
        **kwargs
    ):
        super().__init__(
            learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            name=name,
            **kwargs
        )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)

        self.sparse_steps = []

        for var in var_list:
            if not len(var.shape):
                continue

            if hasattr(var, "path"):
                name = var.path
            else:
                name = str(var.name).replace(":", "_")

            self.sparse_steps.append(
                self.add_variable(
                    shape=(var.shape[:1]) + [1] * (len(var.shape) - 1),
                    initializer="zeros",
                    dtype=var.dtype,
                    name=name.replace("/", "_") + "_sparse_step",
                )
            )

    def update_step(self, gradient, variable, learning_rate):
        if not len(variable.shape) or not isinstance(
            gradient, tf.IndexedSlices
        ):
            return super().update_step(gradient, variable, learning_rate)

        var_idx = self._get_variable_index(variable)
        exp_avg = self.exp_avgs[var_idx]
        exp_avg_sq = self.exp_avg_sqs[var_idx]
        exp_avg_diff = self.exp_avg_diffs[var_idx]
        prev_grad = self.prev_grads[var_idx]
        sparse_step = self.sparse_steps[var_idx]

        exp_avg_t = ops.take(exp_avg, gradient.indices, axis=0)
        exp_avg_sq_t = ops.take(exp_avg_sq, gradient.indices, axis=0)
        exp_avg_diff_t = ops.take(exp_avg_diff, gradient.indices, axis=0)
        prev_grad_t = ops.take(prev_grad, gradient.indices, axis=0)

        sparse_step_t = ops.take(sparse_step, gradient.indices, axis=0)
        local_step_t = sparse_step_t + 1
        gradient_t = ops.cast(gradient.values, variable.dtype)
        prev_grad_t = ops.where(0.0 == sparse_step_t, gradient_t, prev_grad_t)

        exp_avg_t, exp_avg_diff_t, exp_avg_sq_t, var_t = self._update_step(
            variable,
            learning_rate,
            local_step_t,
            gradient_t,
            exp_avg_t,
            exp_avg_sq_t,
            exp_avg_diff_t,
            prev_grad_t,
        )

        exp_avg_t = tf.IndexedSlices(exp_avg_t, gradient.indices)
        exp_avg_diff_t = tf.IndexedSlices(exp_avg_diff_t, gradient.indices)
        exp_avg_sq_t = tf.IndexedSlices(exp_avg_sq_t, gradient.indices)
        var_t = tf.IndexedSlices(var_t, gradient.indices)
        sparse_step_t = tf.IndexedSlices(
            ops.reshape(
                ops.ones_like(gradient.indices, dtype=variable.dtype),
                [-1] + [1] * (len(variable.shape) - 1),
            ),
            gradient.indices,
        )

        self.assign(exp_avg, exp_avg_t)
        self.assign(exp_avg_diff, exp_avg_diff_t)
        self.assign(exp_avg_sq, exp_avg_sq_t)
        self.assign(prev_grad, gradient)
        self.assign_add(variable, var_t)
        self.assign_add(sparse_step, sparse_step_t)


Adan.__doc__ = Adan.__doc__.replace(
    "{{base_optimizer_keyword_args}}", base_optimizer_keyword_args
)
