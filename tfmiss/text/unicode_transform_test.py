import tensorflow as tf

from tfmiss.text.unicode_transform import char_category
from tfmiss.text.unicode_transform import lower_case
from tfmiss.text.unicode_transform import normalize_unicode
from tfmiss.text.unicode_transform import replace_regex
from tfmiss.text.unicode_transform import replace_string
from tfmiss.text.unicode_transform import sub_string
from tfmiss.text.unicode_transform import title_case
from tfmiss.text.unicode_transform import upper_case
from tfmiss.text.unicode_transform import wrap_with
from tfmiss.text.unicode_transform import zero_digits


class CharCategoryTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "z", " "],
            ["❤️", "я", "-"],
        ]
        result = char_category(source)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "z", " "],
            ["❤️", "я", "-"],
        ]
        result = char_category(source)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = char_category("")

        self.assertAllEqual(b"Cn", result)

    def test_0d(self):
        result = char_category("X")

        self.assertAllEqual(b"Lu", result)

    def test_1d(self):
        result = char_category(["X"])

        self.assertAllEqual([b"Lu"], result)

    def test_2d(self):
        result = char_category([["X"]])

        self.assertAllEqual([[b"Lu"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["1", "z"], [" ", "\n", "💥", "❤️"]])
        expected = tf.constant([["Nd", "Ll", "", ""], ["Zs", "Cc", "Cs", "So"]])
        result = char_category(source).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_ragged_last(self):
        source = tf.ragged.constant([["1", "z"], [" ", "\n", "💥", "❤️"]])
        expected = tf.constant([["Nd", "Ll", "", ""], ["Zs", "Cc", "Cs", "Mn"]])
        result = char_category(source, first=False).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = char_category([["X", "-Y-", "z"]], skip=["-Y-"])

        self.assertAllEqual([[b"Lu", b"-Y-", b"Ll"]], result)


class LowerCaseTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = lower_case(source)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = lower_case(source)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = lower_case("")

        self.assertAllEqual(b"", result)

    def test_0d(self):
        result = lower_case("X")

        self.assertAllEqual(b"x", result)

    def test_1d(self):
        result = lower_case(["X"])

        self.assertAllEqual([b"x"], result)

    def test_2d(self):
        result = lower_case([["X"]])

        self.assertAllEqual([[b"x"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["X", "YY"], ["ZZZ ZZZ"]])
        expected = tf.constant([["x", "yy"], ["zzz zzz", ""]])
        result = lower_case(source).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = tf.convert_to_tensor("тест", dtype=tf.string)
        result = lower_case("ТеСт")

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = lower_case([["X", "-Y-", "z"]], skip=["-Y-"])

        self.assertAllEqual([[b"x", b"-Y-", b"z"]], result)


class NormalizeUnicodeTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = normalize_unicode(source, "NFD")

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = normalize_unicode(source, "NFD")
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = normalize_unicode("", "NFD")

        self.assertAllEqual(b"", result)

    def test_0d(self):
        result = normalize_unicode("X", "NFD")

        self.assertAllEqual(b"X", result)

    def test_1d(self):
        result = normalize_unicode(["X"], "NFD")

        self.assertAllEqual([b"X"], result)

    def test_2d(self):
        result = normalize_unicode([["X"]], "NFD")

        self.assertAllEqual([[b"X"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["X", "Y"], ["Z"]])
        expected = tf.constant([["X", "Y"], ["Z", ""]])
        result = normalize_unicode(source, "NFD").to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_n_f_d(self):
        expected = tf.convert_to_tensor("\u0041\u030A", dtype=tf.string)
        result = normalize_unicode("\u00C5", "NFD")

        self.assertAllEqual(expected, result)

    def test_n_f_c(self):
        expected = tf.convert_to_tensor("\u00C5", dtype=tf.string)
        result = normalize_unicode("\u0041\u030A", "NFC")

        self.assertAllEqual(expected, result)

    def test_n_f_k_d(self):
        expected = tf.convert_to_tensor("\u0031", dtype=tf.string)
        result = normalize_unicode("\u2460", "NFKD")

        self.assertAllEqual(expected, result)

    def test_n_f_k_c(self):
        expected = tf.convert_to_tensor("\u1E69", dtype=tf.string)
        result = normalize_unicode("\u1E9B\u0323", "NFKC")

        self.assertAllEqual(expected, result)

    def test_wrong_alg(self):
        with self.assertRaisesRegexp(
            tf.errors.InvalidArgumentError,
            "is not in the list of allowed values",
        ):
            normalize_unicode("", "ABCD")

    def test_skip(self):
        expected = tf.convert_to_tensor(
            [["X", "\u1E9B\u0323", "\u0451"]], dtype=tf.string
        )
        result = normalize_unicode(
            [["X", "\u1E9B\u0323", "\u0435\u0308"]],
            "NFKC",
            skip=["\u1E9B\u0323"],
        )

        self.assertAllEqual(expected, result)


class ReplaceRegexTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = replace_regex(source, ["\\d"], ["0"])

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = replace_regex(source, ["\\d"], ["0"])
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = replace_regex("", ["\\d"], ["0"])

        self.assertAllEqual(b"", result)

    def test_empty_needle(self):
        with self.assertRaisesRegexp(
            tf.errors.InvalidArgumentError,
            'Items of "pattern" could not be empty',
        ):
            result = replace_regex("<test>", [""], [">"])
            self.assertAllEqual(b"test", result)

    def test_empty_haystack(self):
        result = replace_regex("<test>", ["(<)"], [""])

        self.assertAllEqual(b"test>", result)

    def test_0d(self):
        result = replace_regex("1test2", ["\\d"], ["0"])

        self.assertAllEqual(b"0test0", result)

    def test_1d(self):
        result = replace_regex(["1test2"], ["\\d"], ["0"])

        self.assertAllEqual([b"0test0"], result)

    def test_2d(self):
        result = replace_regex([["1test2"]], ["\\d"], ["0"])

        self.assertAllEqual([[b"0test0"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["1test", "test2"], ["test"]])
        expected = tf.constant([["0test", "test0"], ["test", ""]])
        result = replace_regex(source, ["\\d"], ["0"]).to_tensor(
            default_value=""
        )

        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = "_ на _ он же _-0 _ _̈_ _"
        result = replace_regex(
            "тест на юникод он же utf-8 плюс совмещённый символ",
            ["\\pL{3,}", "\\d"],
            ["_", "0"],
        )
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = replace_regex(
            [["1test2", "1t3"]], ["\\d"], ["0"], skip=["1t3"]
        )

        self.assertAllEqual([[b"0test0", b"1t3"]], result)


class ReplaceStringTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = replace_string(source, ["<"], [">"])

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = replace_string(source, ["<"], [">"])
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = replace_string("", ["<"], [">"])

        self.assertAllEqual(b"", result)

    def test_empty_needle(self):
        with self.assertRaisesRegexp(
            tf.errors.InvalidArgumentError,
            'Items of "needle" could not be empty',
        ):
            result = replace_string("<test>", [""], [">"])
            self.assertAllEqual(b"test", result)

    def test_empty_haystack(self):
        result = replace_string("<test>", ["<"], [""])

        self.assertAllEqual(b"test>", result)

    def test_0d(self):
        result = replace_string("<test>", ["<"], [">"])

        self.assertAllEqual(b">test>", result)

    def test_1d(self):
        result = replace_string(["<test>"], ["<"], [">"])

        self.assertAllEqual([b">test>"], result)

    def test_2d(self):
        result = replace_string([["<test>"]], ["<"], [">"])

        self.assertAllEqual([[b">test>"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["<test", "test>"], ["test"]])
        expected = tf.constant([[">test", "test>"], ["test", ""]])
        result = replace_string(source, ["<"], [">"]).to_tensor(
            default_value=""
        )

        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = "тостовый"
        result = replace_string(
            "т́ест", ["́", "е", "ост"], ["", "о", "остовый"]
        )  # noqa: E501
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = replace_string(
            [["<test>", "<unk>"]], ["<"], [">"], skip=["<unk>"]
        )

        self.assertAllEqual([[b">test>", b"<unk>"]], result)


class SubStringTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = sub_string(source, 0, 1)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = sub_string(source, 0, 1)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty_source(self):
        for start in [-1, 0, 1]:
            for limit in [-2, -1, 0, 1, 2]:
                result = sub_string("", start, limit)
                self.assertAllEqual(b"", result)

    def test_empty_sub(self):
        result = sub_string("<test>", 6, 1)

        self.assertAllEqual(b"", result)

    def test_empty_result_left(self):
        result = sub_string("<test>", 0, 0)

        self.assertAllEqual(b"", result)

    def test_empty_result_right(self):
        result = sub_string("<test>", 5, 0)

        self.assertAllEqual(b"", result)

    def test_left_inside(self):
        result = sub_string("<test>", 0, 2)

        self.assertAllEqual(b"<t", result)

    def test_left_last(self):
        result = sub_string("<test>", 0, -1)

        self.assertAllEqual(b"<test>", result)

    def test_left_over(self):
        result = sub_string("<test>", 0, 100)

        self.assertAllEqual(b"<test>", result)

    def test_right_inside(self):
        result = sub_string("<test>", -1, 1)

        self.assertAllEqual(b">", result)

    def test_right_begin(self):
        result = sub_string("<test>", -1, -1)

        self.assertAllEqual(b">", result)

    def test_right_over(self):
        result = sub_string("<test>", -1, 100)

        self.assertAllEqual(b">", result)

    def test_0d(self):
        result = sub_string("<test>", 0, 1)

        self.assertAllEqual(b"<", result)

    def test_1d(self):
        result = sub_string(["<test>"], 0, 1)

        self.assertAllEqual([b"<"], result)

    def test_2d(self):
        result = sub_string([["<test>"]], 0, 1)

        self.assertAllEqual([[b"<"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["<test", "test>"], ["test"]])
        expected = tf.constant([["te", "es"], ["es", ""]])
        result = sub_string(source, 1, 2).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = "с"
        result = sub_string("т́ест", 3, 1)
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = sub_string([["<test>", "<unk>"]], 0, 1, skip=["<unk>"])

        self.assertAllEqual([[b"<", b"<unk>"]], result)


class TitleCaseTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = title_case(source)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = title_case(source)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = title_case("")

        self.assertAllEqual(b"", result)

    def test_0d(self):
        result = title_case("x")

        self.assertAllEqual(b"X", result)

    def test_1d(self):
        result = title_case(["x"])

        self.assertAllEqual([b"X"], result)

    def test_2d(self):
        result = title_case([["x"]])

        self.assertAllEqual([[b"X"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["x", "yy"], ["zzz zzz"]])
        expected = tf.constant([["X", "Yy"], ["Zzz Zzz", ""]])
        result = title_case(source).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_latin(self):
        result = title_case("TeSt")

        self.assertAllEqual(b"Test", result)

    def test_unicode(self):
        expected = ["Тест", "\u01C5"]
        result = title_case(["Тест", "\u01C6"])
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = title_case([["x", "y"]], skip=["y"])

        self.assertAllEqual([[b"X", b"y"]], result)


class UpperCaseUnicodeTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = upper_case(source)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = upper_case(source)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = upper_case("")

        self.assertAllEqual(b"", result)

    def test_0d(self):
        result = upper_case("x")

        self.assertAllEqual(b"X", result)

    def test_1d(self):
        result = upper_case(["x"])

        self.assertAllEqual([b"X"], result)

    def test_2d(self):
        result = upper_case([["x"]])

        self.assertAllEqual([[b"X"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["x", "yy"], ["zzz zzz"]])
        expected = tf.constant([["X", "YY"], ["ZZZ ZZZ", ""]])
        result = upper_case(source).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_latin(self):
        result = upper_case("TeSt")

        self.assertAllEqual(b"TEST", result)

    def test_unicode(self):
        expected = "ТЕСТ"
        result = upper_case("ТеСт")
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = upper_case([["x", "y"]], skip=["y"])

        self.assertAllEqual([[b"X", b"y"]], result)


class WrapWithTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = wrap_with(source, "<", ">")

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = wrap_with(source, "<", ">")
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = wrap_with("", "<", ">")

        self.assertAllEqual(b"<>", result)

    def test_empty_borders(self):
        result = wrap_with("test", "", "")

        self.assertAllEqual(b"test", result)

    def test_0d(self):
        result = wrap_with("X", "<", ">")

        self.assertAllEqual(b"<X>", result)

    def test_1d(self):
        result = wrap_with(["X"], "<", ">")

        self.assertAllEqual([b"<X>"], result)

    def test_2d(self):
        result = wrap_with([["X"]], "<", ">")

        self.assertAllEqual([[b"<X>"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["X", "X"], ["X"]])
        expected = tf.constant([["<X>", "<X>"], ["<X>", ""]])
        result = wrap_with(source, "<", ">").to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = "надо"
        result = wrap_with("ад", "н", "о")
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        self.assertAllEqual(expected, result)

    def test_skip(self):
        result = wrap_with([["X", "y"]], "<", ">", skip=["y"])

        self.assertAllEqual([[b"<X>", b"y"]], result)


class ZeroDigitsTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = zero_digits(source)

        self.assertAllEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
        result = zero_digits(source)
        result = tf.shape(result)

        self.assertAllEqual([2, 3], self.evaluate(result).tolist())

    def test_empty(self):
        result = zero_digits("")

        self.assertEqual(b"", result)

    def test_0d(self):
        result = zero_digits("7")

        self.assertEqual(b"0", result)

    def test_1d(self):
        result = zero_digits(["7"])

        self.assertAllEqual([b"0"], result)

    def test_2d(self):
        result = zero_digits([["7"]])

        self.assertAllEqual([[b"0"]], result)

    def test_ragged(self):
        source = tf.ragged.constant([["x1", "2x"], ["3x4"]])
        expected = tf.constant([["x0", "0x"], ["0x0", ""]])
        result = zero_digits(source).to_tensor(default_value="")

        self.assertAllEqual(expected, result)

    def test_mixed_unicode(self):
        result = zero_digits("P.1, АБ1, ЯК12x, м²")
        expected = tf.convert_to_tensor("P.0, АБ0, ЯК00x, м²", dtype=tf.string)

        self.assertEqual(expected, result)

    def test_skip(self):
        result = zero_digits([["7", "8"]], skip=["8"])

        self.assertAllEqual([[b"0", b"8"]], result)


if __name__ == "__main__":
    tf.test.main()
