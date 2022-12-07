package(default_visibility = ["//visibility:public"])

py_library(
    name = "optimizers",
    srcs = [
        "__init__.py",
        "accum.py",
        "adan.py",
    ],
    data = [
        "//tfmiss/keras/optimizers/schedules",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "accum_test",
    srcs = ["accum_test.py"],
    srcs_version = "PY2AND3",
    deps = [":optimizers"],
)

py_test(
    name = "adan_test",
    srcs = ["adan_test.py"],
    srcs_version = "PY2AND3",
    deps = [":optimizers"],
)
