package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
     "__init__.py",
     "f1.py",
     "bitemp.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "f1_test",
    srcs = ["f1_test.py"],
    srcs_version = "PY2AND3",
    deps = [":losses"],
)

py_test(
    name = "bitemp_test",
    srcs = ["bitemp_test.py"],
    srcs_version = "PY2AND3",
    deps = [":losses"],
)
