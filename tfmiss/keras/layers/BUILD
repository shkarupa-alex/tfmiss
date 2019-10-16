package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "adembed.py",
#        "nce.py",
        "tcn.py",
        "wrappers.py",
    ],
    deps = [
        "//tfmiss/nn",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "adembed_test",
    srcs = ["adembed_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers"
    ],
)

#py_test(
#    name = "nce_test",
#    srcs = ["nce_test.py"],
#    srcs_version = "PY2AND3",
#    deps = [
#        "//tfmiss/keras:testing_utils",
#        ":layers"
#    ],
#)

py_test(
    name = "tcn_test",
    srcs = ["tcn_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

py_test(
    name = "wrappers_test",
    srcs = ["wrappers_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)
