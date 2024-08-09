package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "adan.py",
    ],
    data = [
        "//tfmiss/keras/optimizers/schedules",
    ],
)

py_test(
    name = "adan_test",
    srcs = ["adan_test.py"],
    deps = [":optimizers"],
)
