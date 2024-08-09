package(default_visibility = ["//visibility:public"])

py_library(
    name = "nn",
    srcs = [
        "__init__.py",
        "dcnv2.py",
        "embedding.py",
        "optiact.py",
        "qrnn.py",
    ],
    deps = [
        "//tfmiss/ops",
    ],
)


py_test(
    name = "dcnv2_test",
    size = "large",
    srcs = ["dcnv2_test.py"],
    deps = [":nn"],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    deps = [":nn"],
)

py_test(
    name = "optiact_test",
    srcs = ["optiact_test.py"],
    deps = [":nn"],
)

py_test(
    name = "qrnn_test",
    srcs = ["qrnn_test.py"],
    deps = [":nn"],
)
