from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.metrics import Metric, Precision, Recall
from keras.utils.generic_utils import register_keras_serializable
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

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def get_config(self):
        config = super(F1Binary, self).get_config()
        config.update({'threshold': self.threshold})

        return config
