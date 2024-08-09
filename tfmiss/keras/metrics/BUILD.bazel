package(default_visibility = ["//visibility:public"])

py_library(
    name = "metrics",
    srcs = [
        "__init__.py",
        "f1.py",
    ],
)

py_test(
    name = "f1_test",
    srcs = ["f1_test.py"],
    deps = [":metrics"],
)
