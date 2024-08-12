from collections import Counter

import numpy as np
import tensorflow as tf
from keras.src import backend
from keras.src import testing
from keras.src.saving import register_keras_serializable

from tfmiss.keras.layers.wordvec import CnnEmbedding
from tfmiss.keras.layers.wordvec import Highway
from tfmiss.keras.layers.wordvec import NgramEmbedding
from tfmiss.keras.layers.wordvec import WordEmbedding
from tfmiss.keras.layers.wordvec import WPieceEmbedding


class WordEmbeddingTest(testing.TestCase):
    def test_reserved_words(self):
        layer = WordEmbedding()
        self.assertAllEqual(layer._reserved_words, [layer.UNK_MARK])

        layer = WordEmbedding(reserved_words=["~TesT~"])
        self.assertAllEqual(layer._reserved_words, [layer.UNK_MARK, "~TesT~"])

    def test_merged_vocab(self):
        vocab = ["the", "fox", "jumps", "\u1E69"]
        layer = WordEmbedding(vocab)
        self.assertAllEqual(layer._vocabulary, [layer.UNK_MARK] + vocab)

        layer = WordEmbedding(vocab, reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary, [layer.UNK_MARK, "~TesT~"] + vocab
        )

        layer = WordEmbedding(vocab + ["~TesT~"], reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary, [layer.UNK_MARK, "~TesT~"] + vocab
        )

    def test_build_vocab(self):
        counts = Counter(
            {"the": 4, "fox": 2, "jumps": 2, "\u1E9B\u0323": 2, "dog": 1}
        )
        expected = [
            ("the", 4),
            ("fox", 2),
            ("jumps", 2),
            ("\u1E69", 2),
            ("dog", 1),
        ]

        layer = WordEmbedding()
        self.assertAllEqual(layer.vocab(counts).most_common(), expected)

    def test_adapt_1d(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]

        layer = WordEmbedding()

        result = layer.adapt(data)
        result = backend.convert_to_numpy(result)
        result = np.char.decode(result.astype("S"), "utf-8")

        self.assertAllEqual(result, data)

    def test_adapt_2d(self):
        data = [
            ["[UNK]", "the", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]

        layer = WordEmbedding()

        result = layer.adapt(data)
        result = backend.convert_to_numpy(result)
        result = np.char.decode(result.astype("S"), "utf-8")

        self.assertAllEqual(result, data)

    def test_adapt_ragged(self):
        data = [
            [[b"[UNK]", b"the", b"fox"]],
            [[b"jumps", b"over"], [b"the", b"lazy", b"dog"]],
        ]

        layer = WordEmbedding()

        result = layer.adapt(tf.ragged.constant(data))
        result = result.numpy()

        self.assertAllEqual(result, data)

    def test_adapt_prep_odd(self):
        data = [
            "[UNK]",
            "\u0041\u030A",
            "Fox",
            "jump99",
            "abcdefghik",
            "abcdefghikl",
        ]
        expected = [
            "[UNK]",
            "\u00E5",
            "fox",
            "jump00",
            "abc\uFFFDik",
            "abc\uFFFDkl",
        ]

        layer = WordEmbedding(
            normalize_unicode="NFC",
            lower_case=True,
            zero_digits=True,
            max_len=6,
        )
        result = layer.adapt(data)
        result = backend.convert_to_numpy(result)
        result = np.char.decode(result.astype("S"), "utf-8").tolist()
        self.assertAllEqual(expected, result)

    def test_adapt_prep_even(self):
        data = [
            "[UNK]",
            "\u0041\u030A",
            "Fox",
            "jum99",
            "abcdefghik",
            "abcdefghikl",
        ]
        expected = [
            "[UNK]",
            "\u00E5",
            "fox",
            "jum00",
            "ab\uFFFDik",
            "ab\uFFFDkl",
        ]

        layer = WordEmbedding(
            normalize_unicode="NFC",
            lower_case=True,
            zero_digits=True,
            max_len=5,
        )
        result = layer.adapt(data)
        result = backend.convert_to_numpy(result)
        result = np.char.decode(result.astype("S"), "utf-8").tolist()
        self.assertAllEqual(expected, result)

    def test_preprocess(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        vocab = ["the", "fox", "jumps"]

        layer = WordEmbedding(vocab)

        result = layer.preprocess(data)
        result = backend.convert_to_numpy(result)

        self.assertAllEqual(result, [0, 1, 2, 3, 0, 1, 0, 0])

    def test_layer(self):
        data = np.array(
            ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        )
        vocab = ["the", "fox", "jumps"]

        inputs = WordEmbedding(vocab).preprocess(data)

        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": None,
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": True,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": True,
                "max_len": 4,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": ["[UNK]"] + vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab + ["[UNK]"],
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": ["[UNK]", "~TesT~"],
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_cpu",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "adapt",
                "adapt_cutoff": [2],
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )
        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
                "with_prep": True,
            },
            input_data=tf.constant(data),
            expected_output_dtype="float32",
            expected_output_shape=(8, 12),
        )

    def test_layer_2d(self):
        data = [
            ["[UNK]", "the", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]
        vocab = ["the", "fox", "jumps"]

        inputs = WordEmbedding(vocab).preprocess(data)

        self.run_layer_test(
            WordEmbedding,
            init_kwargs={
                "vocabulary": vocab,
                "output_dim": 12,
                "normalize_unicode": "NFKC",
                "lower_case": False,
                "zero_digits": False,
                "max_len": None,
                "reserved_words": None,
                "embed_type": "dense_auto",
                "adapt_cutoff": None,
                "adapt_factor": 4,
            },
            input_data=inputs,
            expected_output_dtype="float32",
            expected_output_shape=(2, 4, 12),
        )

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_layer_ragged(self):
    #     data = tf.ragged.constant(
    #         [
    #             [["[UNK]", "the", "fox"]],
    #             [["jumps", "over"], ["the", "lazy", "dog"]],
    #         ]
    #     )
    #     vocab = ["the", "fox", "jumps"]
    #
    #     outputs = WordEmbedding(vocab, 5, with_prep=True)(data)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)
    #
    #     inputs = WordEmbedding(vocab).preprocess(data)
    #     outputs = WordEmbedding(vocab, 5)(inputs)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)


@register_keras_serializable(package="Miss")
class NgramEmbeddingWrap(NgramEmbedding):
    def call(self, inputs, **kwargs):
        dense_shape = tf.unstack(tf.shape(inputs))

        row_lengths = []
        for r in range(inputs.shape.rank - 2):
            row_lengths.append(tf.repeat(dense_shape[r + 1], dense_shape[r]))

        val_mask = inputs >= 0
        row_length = tf.reduce_sum(tf.cast(val_mask, "int32"), axis=-1)
        row_length = tf.reshape(row_length, [-1])
        row_lengths.append(row_length)

        outputs = tf.RaggedTensor.from_nested_row_lengths(
            inputs[val_mask], row_lengths
        )
        outputs = super().call(outputs, **kwargs)
        if isinstance(outputs, tf.RaggedTensor):
            outputs = outputs.to_tensor(0.0)
        outputs.set_shape(self.compute_output_shape(inputs.shape))

        return outputs


class NgramEmbeddingTest(testing.TestCase):
    def test_reserved_words(self):
        layer = NgramEmbedding()
        self.assertAllEqual(layer._reserved_words, [layer.UNK_MARK])

        layer = NgramEmbedding(reserved_words=["~TesT~"])
        self.assertAllEqual(layer._reserved_words, [layer.UNK_MARK, "~TesT~"])

    def test_merged_vocab(self):
        vocab = [
            "<th",
            "<the",
            "<the>",
            "he>",
            "the",
            "the>",
            "<fo",
            "<fox",
            "<fox>",
            "<ju",
            "<jum",
            "<jump",
            "<jumps>",
            "<\u1E69>",
            "fox",
            "fox>",
            "jum",
            "jump",
            "jumps",
            "mps",
            "mps>",
            "ox>",
            "ps>",
            "ump",
            "umps",
            "umps>",
        ]
        layer = NgramEmbedding(vocab)
        self.assertAllEqual(layer._vocabulary, [layer.UNK_MARK] + vocab)

        layer = NgramEmbedding(vocab, reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary, [layer.UNK_MARK, "~TesT~"] + vocab
        )

        layer = NgramEmbedding(vocab + ["~TesT~"], reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary, [layer.UNK_MARK, "~TesT~"] + vocab
        )

    def test_build_vocab(self):
        counts = Counter(
            {"the": 4, "fox": 2, "jumps": 2, "\u1E9B\u0323": 2, "dog": 1}
        )
        expected = [
            ("<th", 4),
            ("the", 4),
            ("he>", 4),
            ("<the", 4),
            ("the>", 4),
            ("<the>", 4),
            ("<fo", 2),
            ("fox", 2),
            ("ox>", 2),
            ("<fox", 2),
            ("fox>", 2),
            ("<fox>", 2),
            ("<ju", 2),
            ("jum", 2),
            ("ump", 2),
            ("mps", 2),
            ("ps>", 2),
            ("<jum", 2),
            ("jump", 2),
            ("umps", 2),
            ("mps>", 2),
            ("<jump", 2),
            ("jumps", 2),
            ("umps>", 2),
            ("<jumps>", 2),
            ("<\u1E69>", 2),
            ("<do", 1),
            ("dog", 1),
            ("og>", 1),
            ("<dog", 1),
            ("dog>", 1),
            ("<dog>", 1),
        ]

        layer = NgramEmbedding()
        self.assertAllEqual(layer.vocab(counts).most_common(), expected)

    def test_adapt_1d(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        expected = [
            [b"[UNK]"],
            [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
            [b"<fo", b"fox", b"ox>", b"<fox", b"fox>", b"<fox>"],
            [
                b"<ju",
                b"jum",
                b"ump",
                b"mps",
                b"ps>",
                b"<jum",
                b"jump",
                b"umps",
                b"mps>",
                b"<jump",
                b"jumps",
                b"umps>",
                b"<jumps>",
            ],
            [
                b"<ov",
                b"ove",
                b"ver",
                b"er>",
                b"<ove",
                b"over",
                b"ver>",
                b"<over",
                b"over>",
                b"<over>",
            ],
            [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
            [
                b"<la",
                b"laz",
                b"azy",
                b"zy>",
                b"<laz",
                b"lazy",
                b"azy>",
                b"<lazy",
                b"lazy>",
                b"<lazy>",
            ],
            [b"<do", b"dog", b"og>", b"<dog", b"dog>", b"<dog>"],
        ]

        layer = NgramEmbedding()

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_2d(self):
        data = [
            ["[UNK]", "the", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]
        expected = [
            [
                [b"[UNK]"],
                [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
                [b"<fo", b"fox", b"ox>", b"<fox", b"fox>", b"<fox>"],
                [
                    b"<ju",
                    b"jum",
                    b"ump",
                    b"mps",
                    b"ps>",
                    b"<jum",
                    b"jump",
                    b"umps",
                    b"mps>",
                    b"<jump",
                    b"jumps",
                    b"umps>",
                    b"<jumps>",
                ],
            ],
            [
                [
                    b"<ov",
                    b"ove",
                    b"ver",
                    b"er>",
                    b"<ove",
                    b"over",
                    b"ver>",
                    b"<over",
                    b"over>",
                    b"<over>",
                ],
                [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
                [
                    b"<la",
                    b"laz",
                    b"azy",
                    b"zy>",
                    b"<laz",
                    b"lazy",
                    b"azy>",
                    b"<lazy",
                    b"lazy>",
                    b"<lazy>",
                ],
                [b"<do", b"dog", b"og>", b"<dog", b"dog>", b"<dog>"],
            ],
        ]

        layer = NgramEmbedding()

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_ragged(self):
        data = [
            [[b"[UNK]", b"the", b"fox"]],
            [[b"jumps", b"over"], [b"the", b"lazy", b"dog"]],
        ]
        expected = [
            [
                [
                    [b"[UNK]"],
                    [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
                    [b"<fo", b"fox", b"ox>", b"<fox", b"fox>", b"<fox>"],
                ]
            ],
            [
                [
                    [
                        b"<ju",
                        b"jum",
                        b"ump",
                        b"mps",
                        b"ps>",
                        b"<jum",
                        b"jump",
                        b"umps",
                        b"mps>",
                        b"<jump",
                        b"jumps",
                        b"umps>",
                        b"<jumps>",
                    ],
                    [
                        b"<ov",
                        b"ove",
                        b"ver",
                        b"er>",
                        b"<ove",
                        b"over",
                        b"ver>",
                        b"<over",
                        b"over>",
                        b"<over>",
                    ],
                ],
                [
                    [b"<th", b"the", b"he>", b"<the", b"the>", b"<the>"],
                    [
                        b"<la",
                        b"laz",
                        b"azy",
                        b"zy>",
                        b"<laz",
                        b"lazy",
                        b"azy>",
                        b"<lazy",
                        b"lazy>",
                        b"<lazy>",
                    ],
                    [b"<do", b"dog", b"og>", b"<dog", b"dog>", b"<dog>"],
                ],
            ],
        ]

        layer = NgramEmbedding()

        result = layer.adapt(tf.ragged.constant(data)).numpy()

        self.assertAllEqual(expected, result)

    def test_preprocess(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        vocab = [
            "<th",
            "the",
            "he>",
            "<the",
            "the>",
            "<the>",
            "<fo",
            "fox",
            "ox>",
            "<fox",
            "fox>",
            "<fox>",
            "<ju",
            "jum",
            "ump",
            "mps",
            "ps>",
            "<jum",
            "jump",
            "umps",
            "mps>",
            "<jump",
            "jumps",
            "umps>",
            "<jumps>",
            "<ṩ>",
            "<do",
            "dog",
            "og>",
            "<dog",
            "dog>",
            "<dog>",
        ]
        expected = [
            [0],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [27, 28, 29, 30, 31, 32],
        ]

        layer = NgramEmbedding(vocab)

        result = layer.preprocess(data).numpy()

        self.assertAllEqual(expected, result)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_layer(self):
    #     data = np.array(
    #         ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
    #     )
    #     vocab = [
    #         "<th",
    #         "the",
    #         "he>",
    #         "<the",
    #         "the>",
    #         "<the>",
    #         "<fo",
    #         "fox",
    #         "ox>",
    #         "<fox",
    #         "fox>",
    #         "<fox>",
    #         "<ju",
    #         "jum",
    #         "ump",
    #         "mps",
    #         "ps>",
    #         "<jum",
    #         "jump",
    #         "umps",
    #         "mps>",
    #         "<jump",
    #         "jumps",
    #         "umps>",
    #         "<jumps>",
    #         "<ṩ>",
    #         "<do",
    #         "dog",
    #         "og>",
    #         "<dog",
    #         "dog>",
    #         "<dog>",
    #     ]
    #     inputs = NgramEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         NgramEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": None,
    #             "minn": 2,
    #             "maxn": 5,
    #             "itself": "always",
    #             "reduction": "mean",
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #     self.run_layer_test(
    #         NgramEmbedding,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": None,
    #             "minn": 2,
    #             "maxn": 5,
    #             "itself": "always",
    #             "reduction": "mean",
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #             "with_prep": True,
    #         },
    #         input_data=data,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #
    # def test_layer_2d(self):
    #     data = [
    #         ["[UNK]", "the", "fox", "jumps"],
    #         ["over", "the", "lazy", "dog"],
    #     ]
    #     vocab = [
    #         "<th",
    #         "the",
    #         "he>",
    #         "<the",
    #         "the>",
    #         "<the>",
    #         "<fo",
    #         "fox",
    #         "ox>",
    #         "<fox",
    #         "fox>",
    #         "<fox>",
    #         "<ju",
    #         "jum",
    #         "ump",
    #         "mps",
    #         "ps>",
    #         "<jum",
    #         "jump",
    #         "umps",
    #         "mps>",
    #         "<jump",
    #         "jumps",
    #         "umps>",
    #         "<jumps>",
    #         "<ṩ>",
    #         "<do",
    #         "dog",
    #         "og>",
    #         "<dog",
    #         "dog>",
    #         "<dog>",
    #     ]
    #
    #     inputs = NgramEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         NgramEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": None,
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(2, 4, 12),
    #     )
    #
    # def test_layer_ragged(self):
    #     data = tf.ragged.constant(
    #         [
    #             [["[UNK]", "the", "fox"]],
    #             [["jumps", "over"], ["the", "lazy", "dog"]],
    #         ]
    #     )
    #     vocab = [
    #         "<th",
    #         "the",
    #         "he>",
    #         "<the",
    #         "the>",
    #         "<the>",
    #         "<fo",
    #         "fox",
    #         "ox>",
    #         "<fox",
    #         "fox>",
    #         "<fox>",
    #         "<ju",
    #         "jum",
    #         "ump",
    #         "mps",
    #         "ps>",
    #         "<jum",
    #         "jump",
    #         "umps",
    #         "mps>",
    #         "<jump",
    #         "jumps",
    #         "umps>",
    #         "<jumps>",
    #         "<ṩ>",
    #         "<do",
    #         "dog",
    #         "og>",
    #         "<dog",
    #         "dog>",
    #         "<dog>",
    #     ]
    #
    #     outputs = NgramEmbedding(vocab, 5, with_prep=True)(data)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)
    #
    #     inputs = NgramEmbedding(vocab).preprocess(data)
    #     outputs = NgramEmbedding(vocab, 5)(inputs)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)


@register_keras_serializable(package="Miss")
class WPieceEmbeddingWrap(WPieceEmbedding):
    def call(self, inputs, **kwargs):
        dense_shape = tf.unstack(tf.shape(inputs))

        row_lengths = []
        for r in range(inputs.shape.rank - 2):
            row_lengths.append(tf.repeat(dense_shape[r + 1], dense_shape[r]))

        val_mask = inputs >= 0
        row_length = tf.reduce_sum(tf.cast(val_mask, "int32"), axis=-1)
        row_length = tf.reshape(row_length, [-1])
        row_lengths.append(row_length)

        outputs = tf.RaggedTensor.from_nested_row_lengths(
            inputs[val_mask], row_lengths
        )
        outputs = super().call(outputs, **kwargs)
        if isinstance(outputs, tf.RaggedTensor):
            outputs = outputs.to_tensor(0.0)
        outputs.set_shape(self.compute_output_shape(inputs.shape))

        return outputs


class WPieceEmbeddingTest(testing.TestCase):
    def test_reserved_words(self):
        layer = WPieceEmbedding()
        self.assertAllEqual(
            layer._reserved_words, [layer.UNK_MARK, layer.UNK_CHAR]
        )

        layer = WPieceEmbedding(reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._reserved_words, [layer.UNK_MARK, layer.UNK_CHAR, "~TesT~"]
        )

    def test_merged_vocab(self):
        vocab = [
            "the",
            "##o",
            "f",
            "##u",
            "##m",
            "##s",
            "o",
            "##v",
            "##er",
            "l",
            "##a",
            "##y",
            "d",
            "##g",
        ]
        layer = WPieceEmbedding(vocab)
        self.assertAllEqual(
            layer._vocabulary, [layer.UNK_MARK, layer.UNK_CHAR] + vocab
        )

        layer = WPieceEmbedding(vocab, reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary,
            [layer.UNK_MARK, layer.UNK_CHAR, "~TesT~"] + vocab,
        )

        layer = WPieceEmbedding(vocab + ["~TesT~"], reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary,
            [layer.UNK_MARK, layer.UNK_CHAR, "~TesT~"] + vocab,
        )

    def test_build_vocab(self):
        counts = Counter(
            {"the": 4, "fox": 2, "jumps": 2, "\u1E9B\u0323": 2, "dog": 1}
        )
        expected = [
            ("the", 4),
            ("##o", 3),
            ("f", 2),
            ("##x", 2),
            ("j", 2),
            ("##u", 2),
            ("##m", 2),
            ("##p", 2),
            ("##s", 2),
            ("[UNK]", 2),
            ("d", 1),
            ("##g", 1),
        ]

        layer = WPieceEmbedding(vocab_size=4)
        self.assertAllEqual(layer.vocab(counts).most_common(), expected)

    def test_adapt_1d(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        vocab = [
            "the",
            "##o",
            "f",
            "##u",
            "##m",
            "##s",
            "o",
            "##v",
            "##er",
            "l",
            "##a",
            "##y",
            "d",
            "##g",
        ]
        expected = [
            [b"[UNK]"],
            [b"the"],
            [b"f", b"##o", b"##[UNK]"],
            [b"[UNK]", b"##u", b"##m", b"##[UNK]", b"##s"],
            [b"o", b"##v", b"##er"],
            [b"the"],
            [b"l", b"##a", b"##[UNK]", b"##y"],
            [b"d", b"##o", b"##g"],
        ]

        layer = WPieceEmbedding(vocab)

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_2d(self):
        data = [
            ["[UNK]", "the", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]
        vocab = [
            "the",
            "##o",
            "f",
            "##u",
            "##m",
            "##s",
            "o",
            "##v",
            "##er",
            "l",
            "##a",
            "##y",
            "d",
            "##g",
        ]
        expected = [
            [
                [b"[UNK]"],
                [b"the"],
                [b"f", b"##o", b"##[UNK]"],
                [b"[UNK]", b"##u", b"##m", b"##[UNK]", b"##s"],
            ],
            [
                [b"o", b"##v", b"##er"],
                [b"the"],
                [b"l", b"##a", b"##[UNK]", b"##y"],
                [b"d", b"##o", b"##g"],
            ],
        ]

        layer = WPieceEmbedding(vocab)

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_ragged(self):
        data = [
            [[b"[UNK]", b"the", b"fox"]],
            [[b"jumps", b"over"], [b"the", b"lazy", b"dog"]],
        ]
        vocab = [
            "the",
            "##o",
            "f",
            "##u",
            "##m",
            "##s",
            "o",
            "##v",
            "##er",
            "l",
            "##a",
            "##y",
            "d",
            "##g",
        ]
        expected = [
            [[[b"[UNK]"], [b"the"], [b"f", b"##o", b"##[UNK]"]]],
            [
                [
                    [b"[UNK]", b"##u", b"##m", b"##[UNK]", b"##s"],
                    [b"o", b"##v", b"##er"],
                ],
                [
                    [b"the"],
                    [b"l", b"##a", b"##[UNK]", b"##y"],
                    [b"d", b"##o", b"##g"],
                ],
            ],
        ]

        layer = WPieceEmbedding(vocab)

        result = layer.adapt(tf.ragged.constant(data)).numpy()

        self.assertAllEqual(expected, result)

    def test_preprocess(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        vocab = [
            "the",
            "##o",
            "f",
            "##u",
            "##m",
            "##s",
            "o",
            "##v",
            "##er",
            "l",
            "##a",
            "##y",
            "d",
            "##g",
        ]
        expected = [
            [0],
            [2],
            [4, 3, 1],
            [0, 5, 6, 1, 7],
            [8, 9, 10],
            [2],
            [11, 12, 1, 13],
            [14, 3, 15],
        ]

        layer = WPieceEmbedding(vocab)

        result = layer.preprocess(data).numpy()

        self.assertAllEqual(expected, result)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_layer(self):
    #     data = np.array(
    #         ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
    #     )
    #     vocab = [
    #         "the",
    #         "##o",
    #         "f",
    #         "##u",
    #         "##m",
    #         "##s",
    #         "o",
    #         "##v",
    #         "##er",
    #         "l",
    #         "##a",
    #         "##y",
    #         "d",
    #         "##g",
    #     ]
    #
    #     inputs = WPieceEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         WPieceEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "reduction": "mean",
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #     self.run_layer_test(
    #         WPieceEmbedding,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "reduction": "mean",
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #             "with_prep": True,
    #         },
    #         input_data=data,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #
    # def test_layer_2d(self):
    #     data = [
    #         ["[UNK]", "the", "fox", "jumps"],
    #         ["over", "the", "lazy", "dog"],
    #     ]
    #     vocab = [
    #         "the",
    #         "##o",
    #         "f",
    #         "##u",
    #         "##m",
    #         "##s",
    #         "o",
    #         "##v",
    #         "##er",
    #         "l",
    #         "##a",
    #         "##y",
    #         "d",
    #         "##g",
    #     ]
    #
    #     inputs = WPieceEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         WPieceEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": None,
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(2, 4, 12),
    #     )
    #
    # def test_layer_ragged(self):
    #     data = tf.ragged.constant(
    #         [
    #             [["[UNK]", "the", "fox"]],
    #             [["jumps", "over"], ["the", "lazy", "dog"]],
    #         ]
    #     )
    #     vocab = [
    #         "the",
    #         "##o",
    #         "f",
    #         "##u",
    #         "##m",
    #         "##s",
    #         "o",
    #         "##v",
    #         "##er",
    #         "l",
    #         "##a",
    #         "##y",
    #         "d",
    #         "##g",
    #     ]
    #
    #     outputs = WPieceEmbedding(vocab, 5, with_prep=True)(data)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)
    #
    #     inputs = WPieceEmbedding(vocab).preprocess(data)
    #     outputs = WPieceEmbedding(vocab, 5)(inputs)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)


@register_keras_serializable(package="Miss")
class CnnEmbeddingWrap(CnnEmbedding):
    def call(self, inputs, **kwargs):
        dense_shape = tf.unstack(tf.shape(inputs))

        row_lengths = []
        for r in range(inputs.shape.rank - 2):
            row_lengths.append(tf.repeat(dense_shape[r + 1], dense_shape[r]))

        val_mask = inputs >= 0
        row_length = tf.reduce_sum(tf.cast(val_mask, "int32"), axis=-1)
        row_length = tf.reshape(row_length, [-1])
        row_lengths.append(row_length)

        outputs = tf.RaggedTensor.from_nested_row_lengths(
            inputs[val_mask], row_lengths
        )
        outputs = super().call(outputs, **kwargs)
        if isinstance(outputs, tf.RaggedTensor):
            outputs = outputs.to_tensor(0.0)
        outputs.set_shape(self.compute_output_shape(inputs.shape))

        return outputs


class CnnEmbeddingTest(testing.TestCase):
    def test_reserved_words(self):
        layer = CnnEmbedding()
        self.assertAllEqual(
            layer._reserved_words,
            [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK],
        )

        layer = CnnEmbedding(reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._reserved_words,
            [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK, "~TesT~"],
        )

    def test_merged_vocab(self):
        vocab = ["e", "o", "t", "h"]
        layer = CnnEmbedding(vocab)
        self.assertAllEqual(
            layer._vocabulary,
            [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK] + vocab,
        )

        layer = CnnEmbedding(vocab, reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary,
            [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK, "~TesT~"] + vocab,
        )

        layer = CnnEmbedding(vocab + ["~TesT~"], reserved_words=["~TesT~"])
        self.assertAllEqual(
            layer._vocabulary,
            [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK, "~TesT~"] + vocab,
        )

    def test_build_vocab(self):
        counts = Counter(
            {"the": 4, "fox": 2, "jumps": 2, "\u1E9B\u0323": 2, "dog": 1}
        )
        expected = [
            ("[BOW]", 11),
            ("[EOW]", 11),
            ("t", 4),
            ("h", 4),
            ("e", 4),
            ("o", 3),
            ("f", 2),
            ("x", 2),
            ("j", 2),
            ("u", 2),
            ("m", 2),
            ("p", 2),
            ("s", 2),
            ("\u1E69", 2),
            ("d", 1),
            ("g", 1),
        ]

        layer = CnnEmbedding()
        self.assertAllEqual(layer.vocab(counts).most_common(), expected)

    def test_adapt_1d(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        expected = [
            [b"[BOW]", b"[UNK]", b"[EOW]"],
            [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
            [b"[BOW]", b"f", b"o", b"x", b"[EOW]"],
            [b"[BOW]", b"j", b"u", b"m", b"p", b"s", b"[EOW]"],
            [b"[BOW]", b"o", b"v", b"e", b"r", b"[EOW]"],
            [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
            [b"[BOW]", b"l", b"a", b"z", b"y", b"[EOW]"],
            [b"[BOW]", b"d", b"o", b"g", b"[EOW]"],
        ]

        layer = CnnEmbedding()

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_2d(self):
        data = [
            ["[UNK]", "the", "fox", "jumps"],
            ["over", "the", "lazy", "dog"],
        ]
        expected = [
            [
                [b"[BOW]", b"[UNK]", b"[EOW]"],
                [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
                [b"[BOW]", b"f", b"o", b"x", b"[EOW]"],
                [b"[BOW]", b"j", b"u", b"m", b"p", b"s", b"[EOW]"],
            ],
            [
                [b"[BOW]", b"o", b"v", b"e", b"r", b"[EOW]"],
                [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
                [b"[BOW]", b"l", b"a", b"z", b"y", b"[EOW]"],
                [b"[BOW]", b"d", b"o", b"g", b"[EOW]"],
            ],
        ]

        layer = CnnEmbedding()

        result = layer.adapt(data).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_ragged(self):
        data = [
            [["[UNK]", "the", "fox"]],
            [["jumps", "over"], ["the", "lazy", "dog"]],
        ]
        expected = [
            [
                [
                    [b"[BOW]", b"[UNK]", b"[EOW]"],
                    [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
                    [b"[BOW]", b"f", b"o", b"x", b"[EOW]"],
                ]
            ],
            [
                [
                    [b"[BOW]", b"j", b"u", b"m", b"p", b"s", b"[EOW]"],
                    [b"[BOW]", b"o", b"v", b"e", b"r", b"[EOW]"],
                ],
                [
                    [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
                    [b"[BOW]", b"l", b"a", b"z", b"y", b"[EOW]"],
                    [b"[BOW]", b"d", b"o", b"g", b"[EOW]"],
                ],
            ],
        ]

        layer = CnnEmbedding()

        result = layer.adapt(tf.ragged.constant(data)).numpy()

        self.assertAllEqual(expected, result)

    def test_adapt_prep_odd(self):
        data = [
            "[UNK]",
            "the",
            "fox",
            "0123456789abcdefghij",
            "0123456789abcdefghijk",
        ]
        expected = [
            [b"[BOW]", b"[UNK]", b"[EOW]"],
            [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
            [b"[BOW]", b"f", b"o", b"x", b"[EOW]"],
            [
                b"[BOW]",
                b"0",
                b"1",
                b"2",
                b"3",
                b"\xef\xbf\xbd",
                b"h",
                b"i",
                b"j",
                b"[EOW]",
            ],
            [
                b"[BOW]",
                b"0",
                b"1",
                b"2",
                b"3",
                b"\xef\xbf\xbd",
                b"i",
                b"j",
                b"k",
                b"[EOW]",
            ],
        ]

        layer = CnnEmbedding(max_len=8)
        result = layer.adapt(data).numpy()
        self.assertAllEqual(expected, result)

    def test_adapt_prep_even(self):
        data = [
            "[UNK]",
            "the",
            "fox",
            "0123456789abcdefghij",
            "0123456789abcdefghijk",
        ]
        expected = [
            [b"[BOW]", b"[UNK]", b"[EOW]"],
            [b"[BOW]", b"t", b"h", b"e", b"[EOW]"],
            [b"[BOW]", b"f", b"o", b"x", b"[EOW]"],
            [
                b"[BOW]",
                b"0",
                b"1",
                b"2",
                b"\xef\xbf\xbd",
                b"h",
                b"i",
                b"j",
                b"[EOW]",
            ],
            [
                b"[BOW]",
                b"0",
                b"1",
                b"2",
                b"\xef\xbf\xbd",
                b"i",
                b"j",
                b"k",
                b"[EOW]",
            ],
        ]

        layer = CnnEmbedding(max_len=7)
        result = layer.adapt(data).numpy()
        self.assertAllEqual(expected, result)

    def test_preprocess(self):
        data = ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
        vocab = ["e", "o", "t", "h"]
        expected = [
            [1, 0, 2],
            [1, 5, 6, 3, 2],
            [1, 0, 4, 0, 2],
            [1, 0, 0, 0, 0, 0, 2],
            [1, 4, 0, 3, 0, 2],
            [1, 5, 6, 3, 2],
            [1, 0, 0, 0, 0, 2],
            [1, 0, 4, 0, 2],
        ]

        layer = CnnEmbedding(vocab)

        result = layer.preprocess(data).numpy()

        self.assertAllEqual(expected, result)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_layer(self):
    #     data = np.array(
    #         ["[UNK]", "the", "fox", "jumps", "over", "the", "lazy", "dog"]
    #     )
    #     vocab = ["e", "o", "t", "h"]
    #
    #     inputs = CnnEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         CnnEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "filters": [32, 32, 64, 128, 256, 512, 1024],
    #             "kernels": [1, 2, 3, 4, 5, 6, 7],
    #             "char_dim": 16,
    #             "activation": "tanh",
    #             "highways": 2,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": 50,
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #     self.run_layer_test(
    #         CnnEmbedding,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "filters": [32, 32, 64, 128, 256, 512, 1024],
    #             "kernels": [1, 2, 3, 4, 5, 6, 7],
    #             "char_dim": 16,
    #             "activation": "tanh",
    #             "highways": 2,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": 50,
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #             "with_prep": True,
    #         },
    #         input_data=data,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(8, 12),
    #     )
    #
    # def test_layer_2d(self):
    #     data = [
    #         ["[UNK]", "the", "fox", "jumps"],
    #         ["over", "the", "lazy", "dog"],
    #     ]
    #     vocab = ["e", "o", "t", "h"]
    #
    #     inputs = CnnEmbedding(vocab).preprocess(data)
    #     inputs = inputs.to_tensor(-1)
    #
    #     self.run_layer_test(
    #         CnnEmbeddingWrap,
    #         init_kwargs={
    #             "vocabulary": vocab,
    #             "output_dim": 12,
    #             "normalize_unicode": "NFKC",
    #             "lower_case": False,
    #             "zero_digits": False,
    #             "max_len": None,
    #             "reserved_words": None,
    #             "embed_type": "dense_auto",
    #             "adapt_cutoff": None,
    #             "adapt_factor": 4,
    #         },
    #         input_data=inputs,
    #         expected_output_dtype="float32",
    #         expected_output_shape=(2, 4, 12),
    #     )
    #
    # def test_layer_ragged(self):
    #     data = tf.ragged.constant(
    #         [
    #             [["[UNK]", "the", "fox"]],
    #             [["jumps", "over"], ["the", "lazy", "dog"]],
    #         ]
    #     )
    #     vocab = ["e", "o", "t", "h"]
    #
    #     outputs = CnnEmbedding(vocab, 5, with_prep=True)(data)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)
    #
    #     inputs = CnnEmbedding(vocab).preprocess(data)
    #     outputs = CnnEmbedding(vocab, 5)(inputs)
    #     self.assertLen(outputs, 2)
    #     self.assertLen(outputs[0], 1)
    #     self.assertLen(outputs[0][0], 3)
    #     self.assertLen(outputs[0][0][0], 5)
    #     self.assertLen(outputs[1], 2)
    #     self.assertLen(outputs[1][1], 3)
    #     self.assertLen(outputs[1][1][2], 5)


class HighwayTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Highway,
            init_kwargs={},
            input_shape=(2, 16, 8),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 16, 8),
        )
        self.run_layer_test(
            Highway,
            init_kwargs={},
            input_shape=(2, 16, 8, 4),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 16, 8, 4),
        )


if __name__ == "__main__":
    tf.test.main()
