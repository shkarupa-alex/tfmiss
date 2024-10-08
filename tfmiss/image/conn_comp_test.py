import cv2
import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src.utils import get_file
from keras.src.utils import image_utils

from tfmiss.image.conn_comp import connected_components


class ConnectedComponentsTest(tf.test.TestCase):
    SNAKE = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    ).astype("bool")[None, ..., None]

    def test_zeros(self):
        inputs = np.zeros((100, 20, 50, 3)).astype("bool")

        result = connected_components(inputs)
        self.assertAllEqual(inputs, result)

    def test_ones(self):
        inputs = np.ones((100, 50, 20, 7)).astype("int32")

        result = connected_components(inputs)
        self.assertAllEqual(inputs, result)

    def test_snake(self):
        result = connected_components(self.SNAKE)
        self.assertAllEqual(self.SNAKE, result)

    def test_snake_disconnected(self):
        for i in range(self.SNAKE.shape[1]):
            for j in range(self.SNAKE.shape[2]):

                # If we disconnect any part of the snake except for the
                # endpoints, there will be 2 components.
                if self.SNAKE[0, i, j, 0] and (i, j) not in [(1, 1), (6, 3)]:
                    disconnected_snake = self.SNAKE.copy()
                    disconnected_snake[0, i, j, 0] = 0

                    result = connected_components(disconnected_snake)
                    result = backend.convert_to_numpy(result)
                    self.assertEqual(result.max(), 2)

                    bins = np.bincount(result.ravel())

                    # Nonzero number of pixels labeled 0, 1, or 2.
                    self.assertGreater(bins[0], 0)
                    self.assertGreater(bins[1], 0)
                    self.assertGreater(bins[2], 0)

    def test_conn4(self):
        inputs = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 1, 1, 1, 0, 1, 1],
                [1, 0, 1, 1, 1, 0, 1, 1],
            ]
        )[None, ..., None].astype("int32")
        expected = np.array(
            [
                [1, 1, 1, 0, 0, 0, 2, 2],
                [1, 0, 0, 0, 2, 2, 2, 2],
                [1, 0, 2, 0, 2, 2, 2, 2],
                [1, 0, 2, 2, 2, 2, 2, 2],
                [1, 0, 2, 2, 2, 2, 2, 2],
                [1, 0, 0, 2, 2, 0, 0, 2],
                [0, 2, 2, 2, 2, 0, 2, 2],
                [3, 0, 2, 2, 2, 0, 2, 2],
            ]
        )[None, ..., None].astype("int32")

        result = connected_components(inputs)
        self.assertAllEqual(expected, result)

    def test_no_norm(self):
        inputs = np.random.randint(0, 2, size=(9, 77, 88, 13), dtype="uint8")

        result = connected_components(inputs, normalize=False)
        result = backend.convert_to_numpy(result)
        self.assertLessEqual(result.max(), 77 * 88 * 13 + 1)

    def test_random(self):
        inputs = np.random.randint(0, 2, size=(9, 77, 88, 13), dtype="uint8")

        expected_ = inputs.transpose((0, 3, 1, 2)).reshape((9 * 13, 77, 88))
        expected_ = np.stack(
            [
                cv2.connectedComponents(expected_[i], connectivity=4)[1]
                for i in range(9 * 13)
            ]
        )
        expected = expected_.reshape((9, 13, 77, 88)).transpose((0, 2, 3, 1))

        result = connected_components(inputs)
        self.assertAllEqual(expected, result)

    def test_real(self):
        test_image = get_file(
            "elephant.jpg",
            "https://storage.googleapis.com/tensorflow/"
            "keras.src-applications/tests/elephant.jpg",
        )
        image = image_utils.load_img(
            test_image, target_size=(400, 500), interpolation="bicubic"
        )
        image = image_utils.img_to_array(image)
        image = np.where(image > 127, image, 0).astype("uint8")

        expected = np.stack(
            [
                cv2.connectedComponents(image[..., i], connectivity=4)[1]
                for i in range(image.shape[-1])
            ],
            axis=-1,
        )

        result = connected_components(image[None])[0]
        self.assertAllClose(result, expected)


if __name__ == "__main__":
    tf.test.main()
