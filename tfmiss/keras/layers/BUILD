package(default_visibility = ["//visibility:public"])

py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "tcn.py",
#        "trellis.py",
        "wrappers.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "tcn_test",
    srcs = ["tcn_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)

#py_test(
#    name = "trellis_test",
#    srcs = ["trellis_test.py"],
#    srcs_version = "PY2AND3",
#    deps = [":layers"],
#)

py_test(
    name = "wrappers_test",
    srcs = ["wrappers_test.py"],
    srcs_version = "PY2AND3",
    deps = [":layers"],
)
