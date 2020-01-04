from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.python.platform import tf_logging as logging


def _moving_average(data, window):
    weights = np.ones(window) / window
    data0 = np.repeat(data[0], window - 1)
    data = np.concatenate((data0, data))

    return np.convolve(data, weights, mode='valid')


@tf.keras.utils.register_keras_serializable(package='Miss')
class LRFinder(tf.keras.callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.
    Arguments:
        max_steps: Number of steps to run experiment.
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        metric: Metric to be monitored.
        descending: Should metric be minimized.
        smooth: Parameter for averaging the loss. To pick between 0 and 1.
    Example:
    ```python
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # This callback will stop the training when there is no improvement in
    # the validation loss for three consecutive epochs.
    model.fit(data, labels, epochs=100, callbacks=[callback],
        validation_data=(val_data, val_labels))
    ```
    """

    def __init__(self, max_steps, min_lr=1e-5, max_lr=1e-1, metric='loss', descending=True, smooth=0.95):
        super(LRFinder, self).__init__()

        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.metric = metric
        self.descending = descending
        self.smooth = smooth

        self.monitor_op = np.less if descending else np.greater
        self.curr_step = None
        self.best_value = None
        self.avg_value = None
        self.values = None
        self.lrs = None

    def on_train_begin(self, logs=None):
        self.curr_step = 0
        self.best_value = 0.
        self.avg_value = 0.
        self.values = []
        self.lrs = []

        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.metric)

        # Check preconditions
        if current_value is None:
            self.model.stop_training = True
            logging.warning('LRFinder conditioned on metric `{}` which is not available. '
                            'Available metrics are: {}.'.format(self.metric, ','.join(list(logs.keys()))))
            return
        if np.isnan(current_value) or np.isinf(current_value):
            self.model.stop_training = True
            logging.warning('LRFinder got Nan/Inf metric metric value. Training will be stopped.')
            return

        # Smooth the loss
        self.avg_value = self.smooth * self.avg_value + (1 - self.smooth) * current_value
        smooth_value = self.avg_value / (1 - self.smooth ** (self.curr_step + 1))

        # Check if the loss is not exploding
        if self.curr_step > 0 and self.monitor_op(self.best_value * 5, smooth_value):
            self.model.stop_training = True
            logging.warning('LRFinder found metric explosion. Training will be stopped.')
            return

        # Remember best loss
        if self.monitor_op(smooth_value, self.best_value) or self.curr_step == 0:
            self.best_value = smooth_value

        self.values.append(current_value)

        curr_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.lrs.append(curr_lr)

        if self.curr_step > self.max_steps:
            self.model.stop_training = True
            logging.warning('LRFinder reached final step. Training will be stopped.')
            return

        # Set next lr
        next_lr = self.min_lr + (self.max_lr - self.min_lr) * self.curr_step / self.max_steps
        tf.keras.backend.set_value(self.model.optimizer.lr, next_lr)

        self.curr_step += 1

    def plot(self, average=0):
        if not len(self.values):
            raise ValueError('Observations are empty. Run training first.')

        values = self.values if not average else _moving_average(self.values, average)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.lrs, values)
        plt.xscale('log')
        plt.xlabel('Log(Learning rate)')
        plt.ylabel('Metric ({})'.format(self.metric))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp, format='png')

            return tmp.name

    def find(self, delta=None):
        if not len(self.values):
            raise ValueError('Observations are empty. Run training first.')

        if delta is None:
            one_percent = len(self.values) / 100
            default_delta = [int(i * one_percent) for i in range(1, 11)]
            default_delta = list(set(default_delta))
            default_delta = [d for d in default_delta if d > 1 and d * 2 < len(self.values)]
            default_best = [self.find(d) for d in default_delta]

            return None if not len(default_best) else np.mean(default_best)

        delta = int(delta)
        if delta < 1:
            raise ValueError('Delta should be positive number.')
        if len(self.values) < delta * 2:
            raise ValueError('Delta should be at most twice smaller then number of observed steps.')

        derivatives = []
        for i in range(delta, len(self.values) - delta):
            derivatives.append((self.values[i + delta] - self.values[i - delta]) / (2 * delta))

        best_idx = np.argmax(derivatives) + delta

        return self.lrs[best_idx]

    def get_config(self):
        config = {
            'max_steps': self.max_steps,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'metric': self.metric,
            'descending': self.descending,
            'smooth': self.smooth,
        }

        base_config = super(LRFinder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))