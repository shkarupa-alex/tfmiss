package(default_visibility = ["//visibility:public"])

py_library(
    name = "image",
    srcs = [
        "__init__.py",
        "conn_comp.py",
        "euclidean_dist.py",
    ],
    data = [
        "//tfmiss/ops"
    ],
    srcs_version = "PY2AND3",
)


py_test(
   name = "conn_comp_test",
   srcs = ["conn_comp_test.py"],
   srcs_version = "PY2AND3",
   deps = [":image"],
)

py_test(
   name = "euclidean_dist_test",
   srcs = ["euclidean_dist_test.py"],
   srcs_version = "PY2AND3",
   deps = [":image"],
)
