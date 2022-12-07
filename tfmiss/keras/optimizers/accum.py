import contextlib
import tensorflow as tf
from keras import optimizers
from keras.optimizers.optimizer_v2 import optimizer_v2
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='Miss')
class Accum(tf.__internal__.tracking.DelegatingTrackableMixin, optimizer_v2.OptimizerV2):
    def __init__(self, optimizer, accum_steps, sparse_support=True):
        optimizer = optimizers.get(optimizer)

        if {optimizer.clipnorm, optimizer.global_clipnorm} - {None}:
            raise ValueError('Gradient accumulation is not compatible with `clipnorm` and `global_clipnorm`.')

        if getattr(optimizer, '_is_wrapped_by_grad_accum_optimizer', False):
            raise ValueError('Optimizer is already wrapped by Accum.')
        optimizer._is_wrapped_by_grad_accum_optimizer = True

        self._optimizer = optimizer
        self._accum_steps = accum_steps
        self._sparse_support = sparse_support

        # We don't call super().__init__, since we do not want to call OptimizerV2's constructor.
        tf.__internal__.tracking.DelegatingTrackableMixin.__init__(self, self._optimizer)

    def __dir__(self):
        result = set(super().__dir__())
        if '_optimizer' in result:
            result |= dir(self._optimizer)

        if 'learning_rate' in result:
            result.add('lr')

        return list(result)

    def __getattribute__(self, name):
        if 'lr' == name:
            name = 'learning_rate'

        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            if name == '_optimizer':
                raise e  # Avoid infinite recursion

            try:
                return getattr(self._optimizer, name)
            except AttributeError:
                raise e

    def __setattr__(self, name, value):
        if 'lr' == name:
            name = 'learning_rate'

        try:
            # We cannot check for the 'iterations' attribute as it cannot be set after it is accessed.
            if 'iterations' != name:
                object.__getattribute__(self, name)
            has_attribute = True
        except AttributeError:
            has_attribute = False

        if not has_attribute and '_optimizer' != name and hasattr(self._optimizer, name):
            setattr(self._optimizer, name, value)
        else:
            super().__setattr__(name, value)

    @contextlib.contextmanager
    def scale_iterations(self):
        _ = self.iterations
        backup = self._optimizer._iterations

        try:
            self._optimizer._iterations = backup // 3
            yield
        finally:
            self._optimizer._iterations = backup

    def _create_slots(self, var_list):
        for var in var_list:
            self._optimizer.add_slot(var, 'accum_grad')

            if self._sparse_support and len(var.shape):
                self.add_slot(var, 'accum_step', shape=var.shape[:1])

        self._optimizer._create_slots(var_list)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        accum_apply = self.iterations % self._accum_steps == self._accum_steps - 1
        accum_steps = tf.cast(self._accum_steps, var_dtype)

        apply_state[(var_device, var_dtype)].update(dict(accum_apply=accum_apply, accum_steps=accum_steps))

        with self.scale_iterations():
            self._optimizer._prepare_local(var_device, var_dtype, apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or \
                       self._fallback_apply_state(var_device, var_dtype)

        accum = self._optimizer.get_slot(var, 'accum_grad')

        update = smart_cond(
            coefficients['accum_apply'],
            lambda: self._accum_apply_dense(var, grad, accum, coefficients['accum_steps'], apply_state),
            lambda: accum.assign_add(grad, use_locking=self._use_locking, read_value=False))

        return update

    def _accum_apply_dense(self, var, grad, accum, accum_steps, apply_state):
        accum_t = (accum + grad) * (1. / accum_steps)

        with self.scale_iterations():
            var_update = self._optimizer._resource_apply_dense(accum_t, var, apply_state=apply_state)

        accum_reset = accum.assign(tf.zeros_like(accum), use_locking=self._use_locking, read_value=False)

        return tf.group(var_update, accum_reset)

    def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices, **kwargs):
        if not self._sparse_support:
            raise ValueError('Optimizer not configured to support sparse updates.')

        apply_state = kwargs.get('apply_state', None)

        var_device, var_dtype = handle.device, handle.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or \
                       self._fallback_apply_state(var_device, var_dtype)

        accum = self._optimizer.get_slot(handle, 'accum_grad')
        steps = self._optimizer.get_slot(handle, 'accum_step')

        update = smart_cond(
            coefficients['accum_apply'],
            lambda: self._accum_apply_sparse(handle, grad, indices, accum, steps, apply_state),
            lambda: self._grad_store_sparse(grad, indices, accum, steps))

        return update

    def _accum_apply_sparse(self, var, grad, indices, accum, steps, apply_state):
        accum_t = self._resource_scatter_add(accum, indices, grad)
        steps_t = self._resource_scatter_add(steps, indices, tf.ones_like(indices, dtype=steps.dtype))

        mask_t = steps_t > 0
        accum_t = accum_t[mask_t] * (1. / self._accum_steps)
        indices_t = tf.squeeze(tf.where(mask_t), axis=-1)

        with self.scale_iterations():
            var_update = self._optimizer._resource_apply_sparse_duplicate_indices(
                accum_t, var, indices_t, apply_state=apply_state)

        accum_reset = self._resource_scatter_update(accum, indices_t, tf.zeros_like(accum_t))

        return tf.group(var_update, accum_reset)

    def _grad_store_sparse(self, grad, indices, accum, steps):
        accum_update = self._resource_scatter_add(accum, indices, grad)
        steps_update = self._resource_scatter_add(steps, indices, tf.ones_like(indices, dtype=steps.dtype))

        return tf.group(accum_update, steps_update)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    def get_config(self):
        return {
            'optimizer': optimizers.serialize(self._optimizer),
            'accum_steps': self._accum_steps,
            'sparse_support': self._sparse_support
        }
