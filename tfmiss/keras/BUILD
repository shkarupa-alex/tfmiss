package(default_visibility = ["//visibility:public"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    data = [
        "//tfmiss/keras/callbacks",
        "//tfmiss/keras/layers",
         "//tfmiss/keras/losses",
         "//tfmiss/keras/metrics",
         "//tfmiss/keras/optimizers",
         ":testing_utils"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "testing_utils",
    srcs = [
        "testing_utils.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "testing_utils_test",
    srcs = ["testing_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [":testing_utils"],
)
