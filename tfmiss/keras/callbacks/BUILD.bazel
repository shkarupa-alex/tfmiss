package(default_visibility = ["//visibility:public"])

py_library(
    name = "callbacks",
    srcs = [
        "__init__.py",
        "lrfind.py",
    ],
)

py_test(
    name = "lrfind_test",
    srcs = ["lrfind_test.py"],
    deps = [":callbacks"],
)
