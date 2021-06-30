package(default_visibility = ["//visibility:public"])

py_library(
    name = "text",
    srcs = [
        "__init__.py",
        "unicode_expand.py",
        "unicode_transform.py",
        "wordpiece.py",
        "wordpiecelib.py",
    ],
    data = [
        "//tfmiss/ops"
    ],
    srcs_version = "PY2AND3",
)


py_test(
   name = "unicode_expand_test",
   srcs = ["unicode_expand_test.py"],
   srcs_version = "PY2AND3",
   deps = [":text"],
)

py_test(
   name = "unicode_transform_test",
   srcs = ["unicode_transform_test.py"],
   srcs_version = "PY2AND3",
   deps = [":text"],
)

py_test(
    name = "wordpiece_test",
    srcs = ["wordpiece_test.py"],
    srcs_version = "PY2AND3",
    deps = [":text"],
)

py_test(
    name = "wordpiecelib_test",
    srcs = ["wordpiecelib_test.py"],
    srcs_version = "PY2AND3",
    deps = [":text"],
)
