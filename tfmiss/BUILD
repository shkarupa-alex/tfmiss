package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "tfmiss",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//tfmiss/keras",
        "//tfmiss/nn",
        "//tfmiss/preprocessing",
        "//tfmiss/text",
        "//tfmiss/training",
    ],
    srcs_version = "PY2AND3",
)
