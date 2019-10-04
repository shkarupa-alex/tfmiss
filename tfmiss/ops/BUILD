package(default_visibility = ["//visibility:public"])

load("@local_config_tf//:build_defs.bzl", "D_GLIBCXX_USE_CXX11_ABI")

py_library(
    name = "ops",
    srcs = [
        "__init__.py",
    ],
    data = [
        ":_tfmiss_ops.so",
    ],
    srcs_version = "PY2AND3",
)

cc_binary(
    name = "_tfmiss_ops.so",
    srcs = [
        # TODO: https://github.com/JonathanRaiman/tensorflow_qrnn/blob/master/qrnn.py

        # preprocessing
        "cc/kernels/preprocessing/cont_bow.cc",
        "cc/ops/preprocessing/cont_bow.cc",

        "cc/kernels/preprocessing/sample_mask.cc",
        "cc/ops/preprocessing/sample_mask.cc",

        "cc/kernels/preprocessing/skip_gram.cc",
        "cc/ops/preprocessing/skip_gram.cc",


        # unicode expand
       "cc/kernels/unicode/expand/unicode_expand.h",
       "cc/kernels/unicode/expand/unicode_expand.cc",
       "cc/ops/unicode/expand/expand_shape.h",
       "cc/ops/unicode/expand/expand_shape.cc",

       "cc/kernels/unicode/expand/char_ngrams.cc",
       "cc/ops/unicode/expand/char_ngrams.cc",

       "cc/kernels/unicode/expand/split_chars.cc",
       "cc/ops/unicode/expand/split_chars.cc",

       "cc/kernels/unicode/expand/split_words.cc",
       "cc/ops/unicode/expand/split_words.cc",


        # unicode transform
       "cc/kernels/unicode/transform/unicode_transform.h",
       "cc/kernels/unicode/transform/unicode_transform.cc",

       "cc/kernels/unicode/transform/lower_case.cc",
       "cc/ops/unicode/transform/lower_case.cc",

       "cc/kernels/unicode/transform/normalize_unicode.cc",
       "cc/ops/unicode/transform/normalize_unicode.cc",

       "cc/kernels/unicode/transform/replace_regex.cc",
       "cc/ops/unicode/transform/replace_regex.cc",

       "cc/kernels/unicode/transform/replace_string.cc",
       "cc/ops/unicode/transform/replace_string.cc",

       "cc/kernels/unicode/transform/title_case.cc",
       "cc/ops/unicode/transform/title_case.cc",

       "cc/kernels/unicode/transform/upper_case.cc",
       "cc/ops/unicode/transform/upper_case.cc",

       "cc/kernels/unicode/transform/wrap_with.cc",
       "cc/ops/unicode/transform/wrap_with.cc",

       "cc/kernels/unicode/transform/zero_digits.cc",
       "cc/ops/unicode/transform/zero_digits.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "//third_party/icu/data:custom_data",
        "@icu//:common",
        "@re2//:re2",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        D_GLIBCXX_USE_CXX11_ABI,
    ],
)
