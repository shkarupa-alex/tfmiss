from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.estimator.extenders import add_write_metrics


@test_util.run_all_in_graph_and_eager_modes
class AddWriteMetricsTest(tf.test.TestCase):
    def test_no_error(self):
        TRAIN_URL = 'http://download.tensorflow.org/data/iris_training.csv'
        TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'
        CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        TRAIN_STEPS = 1000
        BATCH_SIZE = 100

        train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
        test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

        def _iris_dataset_from_csv(file_name, batch_size, shuffle):
            dataset = tf.compat.v2.data.experimental.CsvDataset(
                file_name,
                [0.0, 0.0, 0.0, 0.0, 0],
                header=True
            )
            dataset = dataset.batch(batch_size)

            def _transform(f1, f2, f3, f4, labels):
                features = dict(zip(CSV_COLUMN_NAMES, [f1, f2, f3, f4]))

                return features, labels

            dataset = dataset.map(_transform)

            if shuffle:
                dataset = dataset.shuffle(1000)

            return dataset

        feature_columns = [tf.feature_column.numeric_column(key=key) for key in CSV_COLUMN_NAMES]
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

        classifier.train(input_fn=lambda: _iris_dataset_from_csv(train_path, BATCH_SIZE, True), steps=TRAIN_STEPS)
        eval_result = classifier.evaluate(input_fn=lambda: _iris_dataset_from_csv(test_path, BATCH_SIZE, False))

        self.assertIn('my_acc1', eval_result)
        self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc1'])

        if not tf.executing_eagerly() and tf.__version__.startswith('1.'):
            self.assertIn('my_acc2', eval_result)
            self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc2'])

        classifier.predict(input_fn=lambda: _iris_dataset_from_csv(train_path, BATCH_SIZE, False))


if __name__ == "__main__":
    tf.test.main()
