from collections import Counter

import tensorflow as tf

from tfmiss.preprocessing.sampling import down_sample
from tfmiss.preprocessing.sampling import sample_mask
from tfmiss.preprocessing.sampling import sample_probs


class DownSampleTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.constant([], dtype=tf.string)
        freq_vocab = Counter({"_": 1})
        downsampled = down_sample(
            source, freq_vocab, "", 1.0, min_freq=0, seed=1
        )
        self.assertEqual([], self.evaluate(downsampled).tolist())

    def test_dense(self):
        freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        freq_vocab.update(["unk_{}".format(i) for i in range(100)])  # noise

        source = [
            "tensorflow",
            "the",
            "quick",
            "brown",
            "fox",
            "jumped",
            "over",
            "the",
            "lazy",
            "dog",
            "tensorflow",
        ]
        downsampled = down_sample(
            source, freq_vocab, "", 1e-2, min_freq=2, seed=1
        )

        self.assertListEqual(
            [
                b"",
                b"",
                b"quick",
                b"brown",
                b"fox",
                b"",
                b"over",
                b"the",
                b"lazy",
                b"dog",
                b"",
            ],
            self.evaluate(downsampled).tolist(),
        )

    def test_ragged_2d(self):
        freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        freq_vocab.update(["unk_{}".format(i) for i in range(100)])  # noise

        source = tf.ragged.constant(
            [
                ["tensorflow"],
                ["the", "quick", "brown", "fox", "jumped"],
                ["over", "the", "lazy", "dog"],
                ["tensorflow"],
            ]
        )
        downsampled = down_sample(
            source, freq_vocab, "", 1e-2, min_freq=2, seed=1
        ).to_tensor(default_value="")

        self.assertListEqual(
            [
                [b"", b"", b"", b"", b""],
                [b"", b"quick", b"brown", b"fox", b""],
                [b"over", b"the", b"lazy", b"dog", b""],
                [b"", b"", b"", b"", b""],
            ],
            self.evaluate(downsampled).tolist(),
        )

    def test_ragged_3d(self):
        freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        freq_vocab.update(["unk_{}".format(i) for i in range(100)])  # noise

        source = tf.ragged.constant(
            [
                [["tensorflow"]],
                [["the", "quick"], ["brown", "fox", "jumped"]],
                [["over", "the", "lazy", "dog"]],
                [["tensorflow"]],
            ]
        )
        downsampled = down_sample(
            source, freq_vocab, "", 1e-2, min_freq=2, seed=1
        ).to_tensor(default_value="")

        self.assertListEqual(
            [
                [[b"", b"", b"", b""], [b"", b"", b"", b""]],
                [[b"", b"quick", b"", b""], [b"brown", b"fox", b"", b""]],
                [[b"over", b"the", b"lazy", b"dog"], [b"", b"", b"", b""]],
                [[b"", b"", b"", b""], [b"", b"", b"", b""]],
            ],
            self.evaluate(downsampled).tolist(),
        )


class SampleMaskTest(tf.test.TestCase):
    def test_dense_shape_inference(self):
        freq_vocab = Counter({"_": 1})
        mask = sample_mask(["_"], freq_vocab, 1.0, min_freq=0, seed=1)
        self.assertEqual([1], mask.shape.as_list())

    def test_empty(self):
        freq_vocab = Counter({"_": 1})
        mask = sample_mask([], freq_vocab, 1.0, min_freq=0, seed=1)
        self.assertEqual([], self.evaluate(mask).tolist())

    def test_uniq(self):
        freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        freq_vocab.update(["unk_{}".format(i) for i in range(100)])  # noise

        samples = [
            "tensorflow",
            "the",
            "quick",
            "brown",
            "fox",
            "jumped",
            "over",
            "the",
            "lazy",
            "dog",
            "tensorflow",
        ]
        mask = sample_mask(samples, freq_vocab, 1e-2, min_freq=2, seed=1)

        self.assertListEqual(
            [
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
            ],
            self.evaluate(mask).tolist(),
        )

    def test_no_uniq(self):
        freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        freq_vocab.update(["unk_{}".format(i) for i in range(100)])  # noise

        samples = [
            "tensorflow",
            "the",
            "quick",
            "brown",
            "fox",
            "jumped",
            "over",
            "the",
            "lazy",
            "dog",
            "tensorflow",
        ]
        mask = sample_mask(samples, freq_vocab, 1e-2, seed=1)

        self.assertListEqual(
            [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            self.evaluate(mask).tolist(),
        )


class SampleProbsTest(tf.test.TestCase):
    def setUp(self):
        self.freq_vocab = Counter(
            {
                "tensorflow": 99,
                "the": 9,
                "quick": 8,
                "brown": 7,
                "fox": 6,
                "over": 5,
                "dog": 4,
            }
        )
        self.freq_vocab.update(
            ["unk_{}".format(i) for i in range(100)]
        )  # noise

    def test_high(self):
        keys_probs = sample_probs(self.freq_vocab, 0.9)
        self.assertDictEqual(keys_probs, {})

    def test_medium(self):
        keys_probs = sample_probs(self.freq_vocab, 1e-2)
        self.assertDictEqual(
            keys_probs,
            {
                "tensorflow": 0.1790900865309014,
                "the": 0.7786860651291616,
                "quick": 0.8429356057317857,
                "brown": 0.92309518948453,
            },
        )

    def test_low(self):
        keys_probs = sample_probs(self.freq_vocab, 1e-5)
        expected_vocab = {
            "tensorflow": 0.004927141875599405,
            "the": 0.01652619233464507,
            "quick": 0.01754568831066034,
            "brown": 0.018779088914585775,
            "fox": 0.020313158995052875,
            "over": 0.02229342422927143,
            "dog": 0.024987621835300938,
        }
        expected_vocab.update(
            {"unk_{}".format(i): 0.05116524367060188 for i in range(100)}
        )
        self.assertDictEqual(keys_probs, expected_vocab)


if __name__ == "__main__":
    tf.test.main()
