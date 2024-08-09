package(default_visibility = ["//visibility:public"])

py_library(
    name = "training",
    srcs = [
        # TODO: https://github.com/jasonwei20/eda_nlp + change case
        "__init__.py",
        "adapt.py",
        "bucket.py",
    ],
)

py_test(
    name = "adapt_test",
    srcs = ["adapt_test.py"],
    deps = [":training"],
)

py_test(
    name = "bucket_test",
    srcs = ["bucket_test.py"],
    deps = [":training"],
)
