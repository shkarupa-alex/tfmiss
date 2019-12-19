package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
     "__init__.py",
     "bitemp.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "bitemp_test",
    srcs = ["bitemp_test.py"],
    srcs_version = "PY2AND3",
    deps = [":losses"],
)
