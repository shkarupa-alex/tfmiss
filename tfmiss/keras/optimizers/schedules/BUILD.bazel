package(default_visibility = ["//visibility:public"])

py_library(
    name = "schedules",
    srcs = [
        "__init__.py",
        "whca.py",
    ],
)

py_test(
    name = "whca_test",
    srcs = ["whca_test.py"],
    deps = [":schedules"],
)
