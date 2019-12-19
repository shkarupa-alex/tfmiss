from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.estimator.extenders import add_write_metrics


@test_util.run_all_in_graph_and_eager_modes
class AddWriteMetricsTest(tf.test.TestCase):
    def test_no_error(self):
        train_url = 'http://download.tensorflow.org/data/iris_training.csv'
        test_url = 'http://download.tensorflow.org/data/iris_test.csv'
        csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        train_steps = 1000
        batch_size = 100

        train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
        test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

        def _iris_dataset_from_csv(file_name, batch, shuffle):
            dataset = tf.compat.v2.data.experimental.CsvDataset(
                file_name,
                [0.0, 0.0, 0.0, 0.0, 0],
                header=True
            )
            dataset = dataset.batch(batch)

            def _transform(f1, f2, f3, f4, labels):
                features = dict(zip(csv_column_names, [f1, f2, f3, f4]))

                return features, labels

            dataset = dataset.map(_transform)

            if shuffle:
                dataset = dataset.shuffle(1000)

            return dataset

        feature_columns = [tf.feature_column.numeric_column(key=key) for key in csv_column_names]
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

            if not tf.executing_eagerly() and hasattr(tf, 'VERSION'):  # tf v1.x
                tf_acc = tf.metrics.accuracy(labels=labels, predictions=class_ids)
                result['my_acc2'] = tf_acc

            return result

        classifier = add_write_metrics(classifier, metrics_fn)

        classifier.train(input_fn=lambda: _iris_dataset_from_csv(train_path, batch_size, True), steps=train_steps)
        eval_result = classifier.evaluate(input_fn=lambda: _iris_dataset_from_csv(test_path, batch_size, False))

        self.assertIn('my_acc1', eval_result)
        self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc1'])

        if not tf.executing_eagerly() and hasattr(tf, 'VERSION'):  # tf v1.x
            self.assertIn('my_acc2', eval_result)
            self.assertAlmostEqual(eval_result['accuracy'], eval_result['my_acc2'])

        classifier.predict(input_fn=lambda: _iris_dataset_from_csv(train_path, batch_size, False))


if __name__ == "__main__":
    tf.test.main()
