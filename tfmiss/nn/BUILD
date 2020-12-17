package(default_visibility = ["//visibility:public"])

py_library(
    name = "nn",
    srcs = [
        "__init__.py",
        "embedding.py",
        "qrnn.py",
    ],
    deps = [
        "//tfmiss/ops",
    ],
    srcs_version = "PY2AND3",
)


py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    srcs_version = "PY2AND3",
    deps = [":nn"],
)

py_test(
    name = "qrnn_test",
    srcs = ["qrnn_test.py"],
    srcs_version = "PY2AND3",
    deps = [":nn"],
)
