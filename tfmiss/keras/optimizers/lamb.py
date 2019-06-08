from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K

# # TODO https://github.com/btahir/tensorflow-LAMB/blob/master/tensorflow_LAMB.py
# # TODO https://github.com/titu1994/keras-LAMB-Optimizer/blob/master/keras_lamb.py
# class LAMB(keras.optimizers.Adam):
#     """LAMBOptimizer optimizer.
#     Default parameters follow those provided in the original paper.
#     # Arguments
#         lr: float >= 0. Learning rate.
#         beta_1: float, 0 < beta < 1. Generally close to 1.
#         beta_2: float, 0 < beta < 1. Generally close to 1.
#         epsilon: float >= 0. Fuzz factor. If `None`, defaults to 1e-6.
#         weight_decay: float >= 0. Weight decay regularization.
#         decay: float >= 0. Learning rate decay over each update.
#     # References
#         - [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes]
#           (https://arxiv.org/abs/1904.00962)
#     """
#
#     def __init__(self,
#                  learning_rate=0.001,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-7,
#                  weight_decay=0.01,
#                  name='LAMB',
#                  **kwargs):
#         super(LAMB, self).__init__(
#             learning_rate=learning_rate,
#             beta_1=beta_1,
#             beta_2=beta_2,
#             epsilon=epsilon,
#             amsgrad=False,
#             name=name,
#             **kwargs
#         )
#         self.weight_decay = weight_decay
#
#     # def get_updates(self, loss, params):
#     #     grads = self.get_gradients(loss, params)
#     #     self.updates = [K.update_add(self.iterations, 1)]
#     #
#     #     lr = self.lr
#     #     if self.initial_decay > 0:
#     #         lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
#     #                                                   K.dtype(self.decay))))
#     #
#     #     t = K.cast(self.iterations, K.floatx()) + 1
#     #
#     #     ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#     #     vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
#     #     self.weights = [self.iterations] + ms + vs
#     #
#     #     for p, g, m, v in zip(params, grads, ms, vs):
#     #         m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
#     #         v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
#     #
#     #         m_t_hat = m_t / (1. - K.pow(self.beta_1, t))
#     #         v_t_hat = v_t / (1. - K.pow(self.beta_2, t))
#     #
#     #         p_dash = m_t_hat / (K.sqrt(v_t_hat + self.epsilon))
#     #
#     #         if self.weight_decay > 0.:
#     #             wd = self.weight_decay * p
#     #             p_dash = p_dash + wd
#     #
#     #         r1 = K.sqrt(K.sum(K.square(p)))
#     #         r2 = K.sqrt(K.sum(K.square(p_dash)))
#     #
#     #         r = tf.where(tf.greater(r1, 0.),
#     #                      tf.where(tf.greater(r2, 0.),
#     #                               r1 / r2,
#     #                               1.0),
#     #                      1.0)
#     #         # r = r1 / r2
#     #         eta = r * lr
#     #
#     #         p_t = p - eta * p_dash
#     #
#     #         self.updates.append(K.update(m, m_t))
#     #         self.updates.append(K.update(v, v_t))
#     #         new_p = p_t
#     #
#     #         # Apply constraints.
#     #         if getattr(p, 'constraint', None) is not None:
#     #             new_p = p.constraint(new_p)
#     #
#     #         self.updates.append(K.update(p, new_p))
#     #     return self.updates
#
#     def get_config(self):
#         config = {
#             'weight_decay': self.weight_decay
#         }
#         base_config = super(LAMB, self).get_config()
#         del base_config['amsgrad']
#
#         return dict(list(base_config.items()) + list(config.items()))
