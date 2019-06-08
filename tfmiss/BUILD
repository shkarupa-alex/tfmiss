package(default_visibility = ["//visibility:public"])

py_library(
    name = "tfmiss",
    srcs = ([
        "__init__.py",
    ]),
    deps = [
        "//tfmiss/ops:ops",
    ],
    data = [
        ":keras",
        ":preprocessing",
        ":text",
#        ":training",
    ],
    srcs_version = "PY2AND3",
)
