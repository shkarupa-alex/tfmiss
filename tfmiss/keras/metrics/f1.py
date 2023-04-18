from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.metrics import Metric, Precision, Recall
from keras.saving import register_keras_serializable
from tensorflow.python.ops.losses.util import squeeze_or_expand_dimensions


@register_keras_serializable(package='Miss')
class F1Binary(Metric):
    def __init__(self, threshold=0.5, name=None, dtype=None, **kwargs):
        """Creates a `F1Base` instance.

        Args:
            threshold: A float value compared with prediction values to determine the truth value of predictions
                (i.e., above the threshold is `true`, below is `false`).
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Binary, self).__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = threshold

        self.precision = Precision(threshold)
        self.recall = Recall(threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            y_pred, y_true = squeeze_or_expand_dimensions(y_pred, y_true)
        else:
            y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
                y_pred, y_true, sample_weight=sample_weight)

        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        return 2. * tf.math.divide_no_nan(
            precision * recall,
            precision + recall
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super(F1Binary, self).get_config()
        config.update({'threshold': self.threshold})

        return config


@register_keras_serializable(package='Miss')
class F1Micro(F1Binary):
    def __init__(self, name=None, dtype=None, **kwargs):
        """Creates a `F1Macro` instance.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Micro, self).__init__(name=name, dtype=dtype, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]

        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        for c in range(num_classes):
            super(F1Micro, self).update_state(
                tf.cast(y_true == c, y_true.dtype),
                tf.cast(y_pred == c, y_true.dtype),
                sample_weight)

    def get_config(self):
        config = super(F1Binary, self).get_config()
        config.pop('threshold', None)

        return config


@register_keras_serializable(package='Miss')
class F1Macro(Metric):
    def __init__(self, name=None, dtype=None, **kwargs):
        """Creates a `F1Macro` instance.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Macro, self).__init__(name=name, dtype=dtype, **kwargs)

        self.class2f1 = [F1Binary()]

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        for _ in range(num_classes - len(self.class2f1)):
            self.class2f1.append(F1Binary())

        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        for c, f1 in enumerate(self.class2f1):
            f1.update_state(
                tf.cast(y_true == c, y_true.dtype),
                tf.cast(y_pred == c, y_true.dtype),
                sample_weight)

    def result(self):
        return sum([f1.result() for f1 in self.class2f1]) / len(self.class2f1)

    def reset_state(self):
        for f1 in self.class2f1:
            f1.reset_state()
