package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

py_library(
    name = "tfmiss",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//tfmiss/image",
        "//tfmiss/keras",
        "//tfmiss/nn",
        "//tfmiss/preprocessing",
        "//tfmiss/text",
        "//tfmiss/training",
    ],
    srcs_version = "PY2AND3",
)
