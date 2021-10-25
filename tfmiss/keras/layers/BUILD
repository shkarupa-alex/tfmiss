package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "attention.py",
        "dcnv2.py",
        "dropout.py",
        "embedding.py",
        "preprocessing.py",
        "qrnn.py",
        "scale.py",
        "softmax.py",
        "tcn.py",
        "temp.py",
        "todense.py",
        "wordvec.py",
        "wrappers.py",
    ],
    deps = [
        "//tfmiss/nn",
        "//tfmiss/text"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "dcnv2_test",
    srcs = ["dcnv2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "dropout_test",
    srcs = ["dropout_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "preprocessing_test",
    srcs = ["preprocessing_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "qrnn_test",
    srcs = ["qrnn_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tfmiss/keras:testing_utils",
        ":layers"
    ],
)

py_test(
    name = "scale_test",
    srcs = ["scale_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

py_test(
    name = "softmax_test",
    srcs = ["softmax_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tfmiss/keras:testing_utils",
        ":layers"
    ],
)

py_test(
    name = "tcn_test",
    srcs = ["tcn_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

py_test(
    name = "temp_test",
    srcs = ["temp_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

py_test(
    name = "todense_test",
    srcs = ["todense_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

py_test(
    name = "wordvec_test",
    srcs = ["wordvec_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

py_test(
    name = "wrappers_test",
    srcs = ["wrappers_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)
