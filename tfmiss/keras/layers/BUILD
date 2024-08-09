package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "dcnv2.py",
        "dropout.py",
        "embedding.py",
        "preprocessing.py",
        "qrnn.py",
        "reduction.py",
        "scale.py",
        "softmax.py",
        "todense.py",
        "wordvec.py",
        "wrappers.py",
    ],
    deps = [
        "//tfmiss/nn",
        "//tfmiss/text"
    ],
)

py_test(
    name = "dcnv2_test",
    srcs = ["dcnv2_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "dropout_test",
    srcs = ["dropout_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "preprocessing_test",
    srcs = ["preprocessing_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "qrnn_test",
    srcs = ["qrnn_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "reduction_test",
    srcs = ["reduction_test.py"],
    deps = [":layers"],
)

py_test(
    name = "scale_test",
    srcs = ["scale_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "softmax_test",
    srcs = ["softmax_test.py"],
    deps = [
        ":layers"
    ],
)

py_test(
    name = "todense_test",
    srcs = ["todense_test.py"],
    deps = [":layers"],
)

py_test(
    name = "wordvec_test",
    srcs = ["wordvec_test.py"],
    deps = [":layers"],
)

py_test(
    name = "wrappers_test",
    srcs = ["wrappers_test.py"],
    deps = [":layers"],
)
