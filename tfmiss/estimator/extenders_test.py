from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.estimator.extenders import add_write_metrics


@test_util.run_all_in_graph_and_eager_modes
class AddWriteMetricsTest(tf.test.TestCase):
    def testNoError(self):
        TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
        TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
        CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        Y_NAME = 'Species'
        TRAIN_STEPS = 1000
        BATCH_SIZE = 100

        train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
        test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

        train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
        train_x, train_y = train, train.pop(Y_NAME)
        test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
        test_x, test_y = test, test.pop(Y_NAME)

        def train_input_fn(features, labels, batch_size):
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
            dataset = dataset.shuffle(1000).batch(batch_size)

            return dataset

        def eval_input_fn(features, labels, batch_size):
            features = dict(features)
            if labels is None:  # No labels, use only features.
                inputs = features
            else:
                inputs = (features, labels)

            dataset = tf.data.Dataset.from_tensor_slices(inputs)
            dataset = dataset.batch(batch_size)

            return dataset

        feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 10],
            n_classes=3
        )

        def metrics_fn(labels, predictions):
            logits = predictions['logits']
            class_ids = tf.argmax(logits, axis=-1)

            keras_acc = tf.keras.metrics.Accuracy()
            keras_acc.update_state(y_true=labels, y_pred=class_ids)

            result = {'my_acc1': keras_acc}

            if not tf.executing_eagerly() and tf.__version__.startswith('1.'):
                tf_acc = tf.metrics.accuracy(labels=labels, predictions=class_ids)
                result['my_acc2'] = tf_acc

            return result

        classifier = add_write_metrics(classifier, metrics_fn)

        classifier.train(
            input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), steps=TRAIN_STEPS)
        eval_result = classifier.evaluate(
            input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))

        self.assertIn('my_acc1', eval_result)
        self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc1'])

        if not tf.executing_eagerly() and tf.__version__.startswith('1.'):
            self.assertIn('my_acc2', eval_result)
            self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc2'])

        predict_x = {
            'SepalLength': [5.1, 5.9, 6.9],
            'SepalWidth': [3.3, 3.0, 3.1],
            'PetalLength': [1.7, 4.2, 5.4],
            'PetalWidth': [0.5, 1.5, 2.1],
        }

        classifier.predict(input_fn=lambda: eval_input_fn(predict_x, labels=None, batch_size=BATCH_SIZE))


if __name__ == "__main__":
    tf.test.main()
