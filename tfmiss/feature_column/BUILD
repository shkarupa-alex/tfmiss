package(default_visibility = ["//visibility:public"])

py_library(
    name = "feature_column",
    srcs = [
        "__init__.py",
        "embedding.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tfmiss/nn",
        ":feature_column"
    ],
)
