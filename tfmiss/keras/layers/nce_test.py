from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.nce import NoiseContrastiveEstimation


@keras_parameterized.run_all_keras_modes
class NoiseContrastiveEstimationTest(keras_parameterized.TestCase):
    def testModel(self):
        NUM_SAMPLES = 10000
        NUM_CLASSES = 1000000
        EMBED_SIZE = 10
        SAMPLE_SIZE = 1000

        ids = tf.keras.layers.Input(shape=(1,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(1,), dtype='int32')

        item_embedding = tf.keras.layers.Embedding(input_dim=NUM_SAMPLES, output_dim=EMBED_SIZE, input_length=1)
        selected_items = tf.keras.layers.Flatten()(item_embedding(ids))
        h1 = tf.keras.layers.Dense(EMBED_SIZE // 2, activation='relu')(selected_items)
        sm_logits = NoiseContrastiveEstimation(num_classes=NUM_CLASSES, num_negative=100, name='nce')([h1, targets])

        model = tf.keras.Model(inputs=[ids, targets], outputs=sm_logits)
        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )

        Xt = [np.random.randint(NUM_SAMPLES - 1, size=SAMPLE_SIZE), np.ones(SAMPLE_SIZE)]
        Xv = [np.random.randint(NUM_SAMPLES - 1, size=SAMPLE_SIZE // 100), np.ones(SAMPLE_SIZE // 100)]
        Xp = [np.random.randint(NUM_SAMPLES - 1, size=SAMPLE_SIZE // 100), np.zeros(0)]
        model.fit(x=Xt, batch_size=100, epochs=1, validation_data=(Xv, None))
        # model.predict(x=Xp, batch_size=100)


if __name__ == "__main__":
    tf.test.main()
