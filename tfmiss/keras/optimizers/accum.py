import contextlib
import tensorflow as tf
from tf_keras import optimizers
from tf_keras.optimizers import Optimizer
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.control_flow_util import smart_cond


@register_keras_serializable(package='Miss')
class Accum(Optimizer):
    def __init__(self, optimizer, accum_steps, sparse_support=True):
        optimizer = optimizers.get(optimizer)
        if not isinstance(optimizer, Optimizer):
            raise ValueError('Legacy optimizer not supported.')

        if optimizer.global_clipnorm is not None:
            raise ValueError('Gradient accumulation is not compatible with `global_clipnorm`.')

        if getattr(optimizer, '_is_wrapped_by_grad_accum_optimizer', False):
            raise ValueError('Optimizer is already wrapped by Accum.')
        optimizer._is_wrapped_by_grad_accum_optimizer = True

        self._optimizer = optimizer
        self._accum_steps = accum_steps
        self._sparse_support = sparse_support

    def __dir__(self):
        result = super().__dir__()
        if '_optimizer' in result:
            result += dir(self._optimizer)
            result = list(set(result))

        return result

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name == '_optimizer':
                raise e

            try:
                return getattr(self._optimizer, name)
            except AttributeError:
                raise e

    def __setattr__(self, name, value):
        try:
            # We cannot check for the 'iterations' attribute as it cannot be set after it is accessed.
            if 'iterations' != name:
                object.__getattribute__(self, name)
            has_attribute = True
        except AttributeError:
            has_attribute = False

        if not has_attribute and '_optimizer' != name and \
                hasattr(self, '_optimizer') and hasattr(self._optimizer, name):
            setattr(self._optimizer, name, value)
        else:
            super().__setattr__(name, value)

    def build(self, var_list):
        if getattr(self._optimizer, '_built', False):
            return

        self._optimizer.accum_grads = []
        self._optimizer.accum_steps = []

        for var in var_list:
            self._optimizer.accum_grads.append(self._optimizer.add_variable_from_reference(var, 'accum_grad'))

            if self._sparse_support and len(var.shape):
                self._optimizer.accum_steps.append(
                    self._optimizer.add_variable_from_reference(var, 'accum_step', shape=var.shape[:1]))

        self._optimizer.build(var_list)

    def _clip_gradients(self, grads):
        return grads

    def _apply_weight_decay(self, variables):
        if not self.weight_decay:
            return

        accum_apply = self.iterations % self._accum_steps == self._accum_steps - 1
        smart_cond(
            accum_apply,
            lambda: self._optimizer._apply_weight_decay(variables),
            lambda: None)

    def update_step(self, gradient, variable):
        accum_apply = self.iterations % self._accum_steps == self._accum_steps - 1
        accum_steps = tf.cast(self._accum_steps, variable.dtype)

        var_key = self._optimizer._var_key(variable)
        var_idx = self._optimizer._index_dict[var_key]

        accum = self._optimizer.accum_grads[var_idx]

        if isinstance(gradient, tf.IndexedSlices):
            steps = self._optimizer.accum_steps[var_idx]

            smart_cond(
                accum_apply,
                lambda: self._accum_apply_sparse(variable, gradient, accum, steps),
                lambda: self._grad_store_sparse(gradient, accum, steps))

        else:
            smart_cond(
                accum_apply,
                lambda: self._accum_apply_dense(variable, gradient, accum, accum_steps),
                lambda: self._grad_store_dense(gradient, accum))

    @contextlib.contextmanager
    def scale_iterations(self):
        _ = self.iterations
        backup = self._optimizer._iterations

        try:
            self._optimizer._iterations = backup // self._accum_steps
            yield
        finally:
            self._optimizer._iterations = backup

    def _accum_apply_dense(self, var, grad, accum, accum_steps):
        accum_t = (accum + grad) * (1. / accum_steps)
        accum_t = self._optimizer._clip_gradients([accum_t])[0]

        with self.scale_iterations():
            self._optimizer.update_step(accum_t, var)

        accum.assign(tf.zeros_like(accum))

    def _grad_store_dense(self, grad, accum):
        accum.assign_add(grad)

    def _accum_apply_sparse(self, var, grad, accum, steps):
        self._grad_store_sparse(grad, accum, steps)

        mask_t = steps > 0
        values_t = accum[mask_t] * (1. / self._accum_steps)
        indices_t = tf.squeeze(tf.where(mask_t), axis=-1)
        accum_t = tf.IndexedSlices(values_t, indices_t)
        accum_t = self._optimizer._clip_gradients([accum_t])[0]

        with self.scale_iterations():
            self._optimizer.update_step(accum_t, var)

        accum.scatter_update(tf.IndexedSlices(tf.zeros_like(values_t), indices_t))

    def _grad_store_sparse(self, grad, accum, steps):
        steps.scatter_add(tf.IndexedSlices(tf.ones_like(grad.indices, dtype=steps.dtype), grad.indices))
        accum.scatter_add(grad)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer.learning_rate = learning_rate

    @property
    def lr(self):
        return self._optimizer.lr

    @lr.setter
    def lr(self, learning_rate):
        self._optimizer.lr = learning_rate

    def get_config(self):
        return {
            'optimizer': optimizers.serialize(self._optimizer),
            'accum_steps': self._accum_steps,
            'sparse_support': self._sparse_support
        }
