package(default_visibility = ["//visibility:public"])

load("//tfmiss:tfmiss.bzl", "custom_op_library")

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

custom_op_library(
    name = "_tfmiss_ops.so",
    srcs = [
        # dcn v2
        "cc/kernels/dcnv2/dcn_v2.h",
        "cc/kernels/dcnv2/dcn_v2.cc",

        "cc/ops/dcnv2/dcn_v2.cc",


        # preprocessing
        "cc/kernels/preprocessing/cbow_context.cc",
        "cc/ops/preprocessing/cbow_context.cc",

        "cc/kernels/preprocessing/cont_bow.cc",
        "cc/ops/preprocessing/cont_bow.cc",

        "cc/kernels/preprocessing/sample_mask.cc",
        "cc/ops/preprocessing/sample_mask.cc",

        "cc/kernels/preprocessing/skip_gram.cc",
        "cc/ops/preprocessing/skip_gram.cc",

        "cc/kernels/preprocessing/spaces_after.cc",
        "cc/ops/preprocessing/spaces_after.cc",

        "cc/kernels/preprocessing/wordpiece/tokenizer.h",
        "cc/kernels/preprocessing/wordpiece/tokenizer.cc",
        "cc/kernels/preprocessing/wordpiece/word_piece.cc",
        "cc/ops/preprocessing/word_piece.cc",


        # qrnn
        "cc/kernels/qrnn/fo_pool.h",
        "cc/kernels/qrnn/fo_pool.cc",

        "cc/ops/qrnn/fo_pool.cc",


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

        "cc/kernels/unicode/transform/char_category.cc",
        "cc/ops/unicode/transform/char_category.cc",

        "cc/kernels/unicode/transform/lower_case.cc",
        "cc/ops/unicode/transform/lower_case.cc",

        "cc/kernels/unicode/transform/normalize_unicode.cc",
        "cc/ops/unicode/transform/normalize_unicode.cc",

        "cc/kernels/unicode/transform/replace_regex.cc",
        "cc/ops/unicode/transform/replace_regex.cc",

        "cc/kernels/unicode/transform/replace_string.cc",
        "cc/ops/unicode/transform/replace_string.cc",

        "cc/kernels/unicode/transform/sub_string.cc",
        "cc/ops/unicode/transform/sub_string.cc",

        "cc/kernels/unicode/transform/title_case.cc",
        "cc/ops/unicode/transform/title_case.cc",

        "cc/kernels/unicode/transform/upper_case.cc",
        "cc/ops/unicode/transform/upper_case.cc",

        "cc/kernels/unicode/transform/wrap_with.cc",
        "cc/ops/unicode/transform/wrap_with.cc",

        "cc/kernels/unicode/transform/zero_digits.cc",
        "cc/ops/unicode/transform/zero_digits.cc",
    ],
    cuda_srcs = [
        "cc/kernels/dcnv2/dcn_v2.h",
        "cc/kernels/dcnv2/dcn_v2.cu.cc",
        "cc/kernels/qrnn/fo_pool.h",
        "cc/kernels/qrnn/fo_pool.cu.cc",
    ],
    deps = [
        "//third_party/icu/data:custom_data",
        "@icu//:common",
        "@re2//:re2",
    ]
)
