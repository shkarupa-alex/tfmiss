package(default_visibility = ["//visibility:public"])

py_library(
    name = "callbacks",
    srcs = [
        "__init__.py",
        "lrfind.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "lrfind_test",
    srcs = ["lrfind_test.py"],
    srcs_version = "PY2AND3",
    deps = [":callbacks"],
)
