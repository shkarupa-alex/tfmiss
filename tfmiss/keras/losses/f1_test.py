import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models

from tfmiss.keras.losses.f1 import BinarySoftF1
from tfmiss.keras.losses.f1 import MacroSoftF1
from tfmiss.keras.losses.f1 import binary_soft_f1
from tfmiss.keras.losses.f1 import macro_soft_f1


def _to_logit(prob):
    logit = np.log(prob / (1.0 - prob))

    return logit


def _log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))

    return numerator / denominator


class MacroSoftF1Test(tf.test.TestCase):
    def test_config(self):
        bce_obj = MacroSoftF1(reduction="none", name="macro_soft_f1")
        self.assertEqual(bce_obj.name, "macro_soft_f1")
        self.assertEqual(bce_obj.reduction, "none")

    def test_logits(self):
        prediction_tensor = tf.constant(
            [
                [_to_logit(0.97), _to_logit(0.91), _to_logit(0.27)],
                [_to_logit(0.45), _to_logit(0.41), _to_logit(0.05)],
                [_to_logit(0.03), _to_logit(0.97), _to_logit(0.43)],
            ],
            tf.float32,
        )
        target_tensor = tf.constant(
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]], tf.float32
        )

        fl = macro_soft_f1(
            y_true=target_tensor, y_pred=prediction_tensor, from_logits=True
        )

        self.assertAllClose(fl, [0.208014, 0.216742, 0.665546])

    def test_keras_model_compile(self):
        model = models.Sequential(
            [layers.Input(shape=(100,)), layers.Dense(1, activation="sigmoid")]
        )
        model.compile(loss="Miss>macro_soft_f1")

    def test_sigmoid(self):
        prediction_tensor = tf.constant(
            [[0.97, 0.91, 0.27], [0.45, 0.41, 0.05], [0.03, 0.97, 0.43]],
            tf.float32,
        )
        prediction_tensor = tf.nn.sigmoid(prediction_tensor)
        target_tensor = tf.constant(
            [[1, 1, 0], [1, 0, 1], [0, 1, 0]], tf.float32
        )

        fl = macro_soft_f1(y_true=target_tensor, y_pred=prediction_tensor)

        self.assertAllClose(fl, [0.424087, 0.440516, 0.559642], atol=1e-8)


class BinarySoftF1Test(tf.test.TestCase):
    def test_config(self):
        bce_obj = BinarySoftF1(reduction="none", name="binary_soft_f1")
        self.assertEqual(bce_obj.name, "binary_soft_f1")
        self.assertEqual(bce_obj.reduction, "none")

    def test_logits(self):
        prediction_tensor = tf.constant(
            [
                [_to_logit(0.97)],
                [_to_logit(0.45)],
                [_to_logit(0.03)],
            ],
            tf.float32,
        )
        target_tensor = tf.constant([[1], [1], [0]], tf.float32)

        fl = binary_soft_f1(
            y_true=target_tensor, y_pred=prediction_tensor, from_logits=True
        )

        self.assertAllClose(fl, [0.507614, 0.689655, 0.507614])

    def test_keras_model_compile(self):
        model = models.Sequential(
            [layers.Input(shape=(100,)), layers.Dense(1, activation="sigmoid")]
        )
        model.compile(loss="Miss>binary_soft_f1")

    def test_sigmoid(self):
        prediction_tensor = tf.constant([[0.97], [0.45], [0.03]], tf.float32)
        prediction_tensor = tf.nn.sigmoid(prediction_tensor)
        target_tensor = tf.constant([[1], [1], [0]], tf.float32)

        fl = binary_soft_f1(y_true=target_tensor, y_pred=prediction_tensor)

        self.assertAllClose(fl, [0.57967, 0.620872, 0.670017], atol=1e-8)


if __name__ == "__main__":
    tf.test.main()
