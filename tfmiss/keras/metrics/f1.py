from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.backend import batch_set_value


class F1Base(Metric):
    def __init__(self, num_classes, thresholds=None, top_k=None, name=None, dtype=None):
        """Creates a `F1Base` instance.

        Args:
            num_classes: Total number of classes. I.e. the last dimension of predictions.
            thresholds: (Optional) A float value or a python list/tuple of float threshold values in [0, 1].
                A threshold is compared with prediction values to determine the truth value of predictions
                (i.e., above the threshold is `true`, below is `false`). One metric value is generated
                for each threshold value. If neither thresholds nor top_k are set, the default is to calculate
                score with `thresholds=0.5`.
            top_k: (Optional) Unset by default. An int value specifying the top-k predictions to consider
                when calculating score.
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super(F1Base, self).__init__(name=name, dtype=dtype)

        if num_classes < 2:
            raise ValueError('Wrong number of classes: {}'.format(num_classes))

        self.num_classes = num_classes
        self.init_thresholds = thresholds
        self.top_k = top_k

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(thresholds, default_threshold=default_threshold)

        self.num_slices = num_classes if num_classes > 2 else 1
        for i in range(self.num_slices):
            true_positives = self.add_weight(
                'true_positives_{}'.format(i),
                shape=(len(self.thresholds),),
                initializer=tf.keras.initializers.zeros)
            false_positives = self.add_weight(
                'false_positives_{}'.format(i),
                shape=(len(self.thresholds),),
                initializer=tf.keras.initializers.zeros)
            false_negatives = self.add_weight(
                'false_negatives_{}'.format(i),
                shape=(len(self.thresholds),),
                initializer=tf.keras.initializers.zeros)

            setattr(self, 'true_positives_{}'.format(i), true_positives)
            setattr(self, 'false_positives_{}'.format(i), false_positives)
            setattr(self, 'false_negatives_{}'.format(i), false_negatives)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive, false positive and false negative statistics.

        Args:
            y_true: The ground truth values, with the same dimensions as `y_pred`. Will be cast to `bool`.
            y_pred: The predicted values. Each element must be in the range `[0, 1]`.
            sample_weight: Optional weighting of each example. Defaults to 1. Can be a `Tensor` whose rank
                is either 0, or the same rank as `y_true`, and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        update_ops = []
        for i in range(self.num_slices):
            true_positives = getattr(self, 'true_positives_{}'.format(i))
            false_positives = getattr(self, 'false_positives_{}'.format(i))
            false_negatives = getattr(self, 'false_negatives_{}'.format(i))
            class_id = None if self.num_classes == 2 else i

            update_ops.append(metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES: true_positives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES: false_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: false_negatives
                },
                y_true,
                y_pred,
                thresholds=self.thresholds,
                top_k=self.top_k,
                class_id=class_id,
                sample_weight=sample_weight
            ))

        return tf.group(update_ops) if len(update_ops) > 1 else update_ops[0]

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
        }
        base_config = super(F1Base, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class F1Binary(F1Base):
    def __init__(self, thresholds=None, top_k=None, name=None, dtype=None):
        """Creates a `F1Binary` instance."""
        super(F1Binary, self).__init__(num_classes=2, thresholds=thresholds, top_k=top_k, name=name, dtype=dtype)

    def result(self):
        precision = tf.math.divide_no_nan(self.true_positives_0, self.true_positives_0 + self.false_positives_0)
        recall = tf.math.divide_no_nan(self.true_positives_0, self.true_positives_0 + self.false_negatives_0)

        result = 2. * tf.math.divide_no_nan(
            precision * recall,
            precision + recall
        )

        return result[0] if len(self.thresholds) == 1 else result

    @classmethod
    def from_config(cls, config):
        config.pop('num_classes', None)

        return cls(**config)


class F1Macro(F1Base):
    def __init__(self, *args, **kwargs):
        """Creates a `F1Macro` instance."""
        super(F1Macro, self).__init__(*args, **kwargs)

        if 2 == self.num_classes:
            tf.logging.warning('Consider using "F1Binary" metric for F1-score with 2 classes.')

    def result(self):
        precisions, recalls = [], []

        for i in range(self.num_slices):
            _true_positives = getattr(self, 'true_positives_{}'.format(i))
            _false_positives = getattr(self, 'false_positives_{}'.format(i))
            _false_negatives = getattr(self, 'false_negatives_{}'.format(i))

            _precision = tf.math.divide_no_nan(_true_positives, _true_positives + _false_positives)
            _recall = tf.math.divide_no_nan(_true_positives, _true_positives + _false_negatives)

            precisions.append(_precision)
            recalls.append(_recall)

        precision = tf.div(
            tf.add_n(precisions),
            self.num_classes
        )
        recall = tf.div(
            tf.add_n(recalls),
            self.num_classes
        )

        result = 2. * tf.math.divide_no_nan(
            precision * recall,
            precision + recall
        )

        return result[0] if len(self.thresholds) == 1 else result


class F1Micro(F1Base):
    def __init__(self, *args, **kwargs):
        """Creates a `F1Micro` instance."""
        super(F1Micro, self).__init__(*args, **kwargs)

        if 2 == self.num_classes:
            tf.logging.warning('Consider using "F1Binary" metric for F1-score with 2 classes.')

    def result(self):
        true_positives, false_positives, false_negatives = [], [], []

        for i in range(self.num_slices):
            true_positives.append(getattr(self, 'true_positives_{}'.format(i)))
            false_positives.append(getattr(self, 'false_positives_{}'.format(i)))
            false_negatives.append(getattr(self, 'false_negatives_{}'.format(i)))

        precision = tf.math.divide_no_nan(
            tf.add_n(true_positives),
            tf.add_n(true_positives + false_positives),
        )
        recall = tf.math.divide_no_nan(
            tf.add_n(true_positives),
            tf.add_n(true_positives + false_negatives),
        )

        result = 2. * tf.math.divide_no_nan(
            precision * recall,
            precision + recall
        )

        return result[0] if len(self.thresholds) == 1 else result
