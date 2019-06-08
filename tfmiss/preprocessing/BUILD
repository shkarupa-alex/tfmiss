package(default_visibility = ["//visibility:public"])

py_library(
    name = "preprocessing",
    srcs = ([
        "__init__.py",
        "preprocessing.py",
        "sampling.py",
    ]),
    deps = [
        "//tfmiss/ops",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "preprocessing_test",
    srcs = ["preprocessing_test.py"],
    srcs_version = "PY2AND3",
    deps = [":preprocessing"],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    srcs_version = "PY2AND3",
    deps = [":preprocessing"],
)
