package(default_visibility = ["//visibility:public"])

py_library(
    name = "feature_column",
    srcs = [
        "__init__.py",
        "adembed.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "adembed_test",
    srcs = ["adembed_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tfmiss/nn",
        ":feature_column"
    ],
)
