from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tf_keras import utils
from scipy.ndimage import distance_transform_edt
from tensorflow.python.framework import test_util
from tfmiss.image.euclidean_dist import euclidean_distance


@test_util.run_all_in_graph_and_eager_modes
class EuclideanDistanceTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters([
        'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'half', 'bfloat16', 'float',
        'double'])
    def test_zeros(self, idtype):
        inputs = np.zeros([100, 20, 50, 3], dtype=idtype)
        expected = np.zeros([100, 20, 50, 3], dtype='float32')

        result = euclidean_distance(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, expected)

    @parameterized.parameters([
        'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'half', 'bfloat16', 'float',
        'double'])
    def test_ones(self, idtype):
        inputs = np.ones([100, 50, 20, 7], dtype=idtype)
        expected = np.ones([100, 50, 20, 7], dtype='float32') * tf.float32.max

        result = euclidean_distance(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, expected)

    @parameterized.parameters([
        'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'half', 'bfloat16', 'float',
        'double'])
    def test_single(self, idtype):
        inputs = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]], dtype=idtype)[None, ..., None]
        expected = np.array([
            [2, 2.23606801, 2, 2.23606801, 2],
            [1, 1.41421354, 1, 1.41421354, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0]], dtype='float32')[None, ..., None]

        result = euclidean_distance(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, expected)

    @parameterized.parameters([
        'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'half', 'bfloat16', 'float',
        'double'])
    def test_batch(self, idtype):
        batch_size = 3
        inputs = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype=idtype)[None, ..., None].repeat(batch_size, 0)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype='float32')[None, ..., None].repeat(batch_size, 0)

        result = euclidean_distance(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, expected)

    @parameterized.parameters([
        'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'half', 'bfloat16', 'float',
        'double'])
    def test_channels(self, idtype):
        channel_size = 3
        inputs = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype=idtype)[None, ..., None].repeat(channel_size, -1)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype='float32')[None, ..., None].repeat(channel_size, -1)

        result = euclidean_distance(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, expected)

    def test_real(self):
        file = tf.keras.utils.get_file(
            'grace_hopper.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
        image = tf.keras.utils.load_img(file, target_size=[400, 500])
        image = utils.img_to_array(image)
        image = np.where(image > 127, 255, 0)

        expected = np.stack([distance_transform_edt(image[..., i]) for i in range(image.shape[-1])], axis=-1)

        result = euclidean_distance(image[None])[0]
        result = self.evaluate(result)
        self.assertAllClose(result, expected)


if __name__ == '__main__':
    tf.test.main()
