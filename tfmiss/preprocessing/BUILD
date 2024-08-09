package(default_visibility = ["//visibility:public"])

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
        "preprocessing.py",
        "sampling.py",
    ],
    deps = [
        "//tfmiss/ops",
    ],
)

py_test(
    name = "preprocessing_test",
    srcs = ["preprocessing_test.py"],
    deps = [
        "//tfmiss/text",
        ":preprocessing"
    ],
)

py_test(
    name = "sampling_test",
    srcs = ["sampling_test.py"],
    deps = [":preprocessing"],
)
