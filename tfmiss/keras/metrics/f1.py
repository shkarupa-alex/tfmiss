import tensorflow as tf
from keras.src.metrics import Metric
from keras.src.metrics import Precision
from keras.src.metrics import Recall
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Miss")
class F1Binary(Metric):
    def __init__(self, threshold=0.5, name=None, dtype=None):
        """Creates a `F1Base` instance.

        Args:
            threshold: A float value compared with prediction values to
              determine the truth value of predictions (i.e., above the
              threshold is `true`, below is `false`).
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Binary, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold

        self.precision = Precision(threshold)
        self.recall = Recall(threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()

        return 2.0 * tf.math.divide_no_nan(
            precision * recall, precision + recall
        )

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def get_config(self):
        config = super(F1Binary, self).get_config()
        config.update({"threshold": self.threshold})

        return config


@register_keras_serializable(package="Miss")
class F1Micro(F1Binary):
    def __init__(self, name=None, dtype=None):
        """Creates a `F1Macro` instance.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Micro, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]

        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        for c in range(num_classes):
            super(F1Micro, self).update_state(
                tf.cast(y_true == c, y_true.dtype),
                tf.cast(y_pred == c, y_true.dtype),
                sample_weight,
            )

    def get_config(self):
        config = super(F1Binary, self).get_config()
        config.pop("threshold", None)

        return config


@register_keras_serializable(package="Miss")
class F1Macro(Metric):
    def __init__(self, name=None, dtype=None):
        """Creates a `F1Macro` instance.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Macro, self).__init__(name=name, dtype=dtype)

        self.class2f1 = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]

        if not self.class2f1:
            for i in range(num_classes):
                self.class2f1.append(F1Binary())

        y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
        for c, f1 in enumerate(self.class2f1):
            f1.update_state(
                tf.cast(y_true == c, y_true.dtype),
                tf.cast(y_pred == c, y_true.dtype),
                sample_weight,
            )

    def result(self):
        return sum([f1.result() for f1 in self.class2f1]) / len(self.class2f1)

    def reset_state(self):
        for f1 in self.class2f1:
            f1.reset_state()
