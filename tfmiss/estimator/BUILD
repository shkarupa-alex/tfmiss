package(default_visibility = ["//visibility:public"])

py_library(
    name = "estimator",
    srcs = [
        "__init__.py",
        "extenders.py"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "extenders_test",
    srcs = ["extenders_test.py"],
    srcs_version = "PY2AND3",
    deps = [":estimator"],
)