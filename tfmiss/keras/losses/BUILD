package(default_visibility = ["//visibility:public"])

py_library(
    name = "losses",
    srcs = [
     "__init__.py",
     "f1.py",
     "bitemp.py",
    ],
)

py_test(
    name = "f1_test",
    srcs = ["f1_test.py"],
    deps = [":losses"],
)

py_test(
    name = "bitemp_test",
    srcs = ["bitemp_test.py"],
    deps = [":losses"],
)
