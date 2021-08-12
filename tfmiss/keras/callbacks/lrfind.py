from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tempfile
from keras import backend, callbacks
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='Miss')
class LRFinder(callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        max_steps: Number of steps to run experiment.
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        smooth: Parameter for averaging the loss. Pick between 0 and 1.
    """

    def __init__(self, max_steps, min_lr=1e-7, max_lr=10., smooth=0.98):
        super(LRFinder, self).__init__()

        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smooth = smooth

        self.stop_training = None
        self.curr_step = None
        self.best_value = None
        self.avg_loss = None
        self.losses = None
        self.lrs = None

    def on_train_begin(self, logs=None):
        self.stop_training = False
        self.curr_step = 0
        self.best_value = 0.
        self.avg_loss = 0.
        self.losses = []
        self.lrs = []

        backend.set_value(self.model.optimizer.lr, self.min_lr)
        tf.get_logger().warning('Don\'t forget to set "epochs=1" and "steps_per_epoch={}" '
                                'in model.fit() call'.format(self.max_steps))

    def on_train_batch_end(self, batch, logs=None):
        if self.stop_training:
            return

        logs = logs or {}
        current_loss = logs.get('loss', None)

        # Check preconditions
        if current_loss is None:
            self.stop_training = True
            self.model.stop_training = True
            tf.get_logger().error('LRFinder conditioned on "loss" which is not available. '
                                  'Training will be stopped.')
            return
        if np.isnan(current_loss) or np.isinf(current_loss):
            self.stop_training = True
            self.model.stop_training = True
            tf.get_logger().error('LRFinder got Nan/Inf loss value. Training will be stopped.')
            return

        # Smooth the loss
        self.avg_loss = self.smooth * self.avg_loss + (1 - self.smooth) * current_loss
        smooth_loss = self.avg_loss / (1 - self.smooth ** (self.curr_step + 1))

        # Check if the loss is not exploding
        if self.curr_step > 0 and smooth_loss > self.best_value * 4:
            self.stop_training = True
            self.model.stop_training = True
            tf.get_logger().error('LRFinder found loss explosion. Training will be stopped.')
            return

        # Remember best loss
        if self.curr_step == 0 or smooth_loss < self.best_value:
            self.best_value = smooth_loss

        curr_lr = backend.get_value(self.model.optimizer.lr)
        self.lrs.append(curr_lr)
        self.losses.append(smooth_loss)

        if self.curr_step > self.max_steps:
            self.stop_training = True
            self.model.stop_training = True
            tf.get_logger().info('LRFinder reached final step. Training will be stopped.')
            return

        # Set next lr (annealing exponential)
        next_lr = self.min_lr * (self.max_lr / self.min_lr) ** (self.curr_step / self.max_steps)
        backend.set_value(self.model.optimizer.lr, next_lr)

        self.curr_step += 1

    def on_train_end(self, logs=None):
        if self.curr_step < self.max_steps:
            tf.get_logger().error('LRFinder finished before "max_steps" reached. Run training with more data.')
        else:
            tf.get_logger().info('LRFinder finished. Use ".plot()" method see the graph.')

    def plot(self, skip_start=10, skip_end=5):
        if not len(self.losses):
            raise ValueError('Observations are empty. Run training first.')

        losses = self.losses[skip_start:-skip_end]
        lrs = self.lrs[skip_start:-skip_end]

        try:
            min_grad = np.gradient(losses).argmin()
        except ValueError:
            raise ValueError('Failed to compute gradients, there might not be enough points.')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Log(Learning rate), best {:.2e}'.format(lrs[min_grad]))
        plt.ylabel('Loss')
        plt.plot(lrs[min_grad], losses[min_grad], markersize=10, marker='o', color='red')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp, format='png')
            tf.get_logger().info('Graph saved to {}'.format(tmp.name))

            return lrs[min_grad], tmp.name

    def get_config(self):
        config = super(LRFinder, self).get_config()
        config.update({
            'max_steps': self.max_steps,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'smooth': self.smooth,
        })

        return config
