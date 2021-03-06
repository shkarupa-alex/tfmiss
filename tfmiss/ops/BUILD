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
        # preprocessing
        "cc/kernels/preprocessing/cbow_context.cc",
        "cc/ops/preprocessing/cbow_context.cc",

        "cc/kernels/preprocessing/cont_bow.cc",
        "cc/ops/preprocessing/cont_bow.cc",

        "cc/kernels/preprocessing/sample_mask.cc",
        "cc/ops/preprocessing/sample_mask.cc",

        "cc/kernels/preprocessing/skip_gram.cc",
        "cc/ops/preprocessing/skip_gram.cc",


        # qrnn
        "cc/kernels/qrnn/fo_pool.h",
        "cc/kernels/qrnn/fo_pool.cc",

        "cc/ops/qrnn/fo_pool_batch_bwd.cc",
        "cc/ops/qrnn/fo_pool_time_bwd.cc",
        "cc/ops/qrnn/fo_pool_batch.cc",
        "cc/ops/qrnn/fo_pool_time.cc",


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
        "cc/kernels/qrnn/fo_pool.h",
        "cc/kernels/qrnn/fo_pool.cu.cc",
    ],
    deps = [
        "//third_party/icu/data:custom_data",
        "@icu//:common",
        "@re2//:re2",
    ]
)
