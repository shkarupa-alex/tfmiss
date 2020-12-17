package(default_visibility = ["//visibility:public"])

py_library(
    name = "schedules",
    srcs = [
        "__init__.py",
        "whca.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "whca_test",
    srcs = ["whca_test.py"],
    srcs_version = "PY2AND3",
    deps = [":schedules"],
)
