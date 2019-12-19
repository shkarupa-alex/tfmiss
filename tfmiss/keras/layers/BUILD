package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        # TODO: https://mlwhiz.com/blog/2018/12/17/text_classification/
        # TODO: https://github.com/CyberZHG/keras-self-attention/
        "__init__.py",
        "embedding.py",
        "softmax.py",
        "tcn.py",
        "wrappers.py",
    ],
    deps = [
        "//tfmiss/nn",
    ],
    srcs_version = "PY2AND3",
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
    name = "wrappers_test",
    srcs = ["wrappers_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)
