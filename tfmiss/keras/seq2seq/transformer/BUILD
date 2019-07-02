package(default_visibility = ["//visibility:public"])

py_library(
    name = "transformer",
    srcs = [
        "__init__.py",
        "attention.py",
        "position.py",
    ],
    deps = [
        "//tfmiss/training",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":transformer",
        "//tfmiss/keras:testing_utils",
    ],
)

py_test(
    name = "position_test",
    srcs = ["position_test.py"],
    srcs_version = "PY2AND3",
    deps = [":transformer"],
)
