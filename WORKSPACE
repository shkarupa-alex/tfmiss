load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/tf:tf_configure.bzl", "tf_configure")
load("//third_party/gpu:cuda_configure.bzl", "cuda_configure")

# curl -L https://github.com/.../.../archive/<git hash>.tar.gz | [g]sha256sum

http_archive(
    name = "icu",
    sha256 = "65271a83fa81783d1272553f4564965ac2e32535a58b0b8141e9f4003afb0e3a",
    strip_prefix = "icu-release-64-2",
    urls = ["https://github.com/unicode-org/icu/archive/release-64-2.tar.gz"],
    build_file = "//third_party/icu:BUILD.bzl",
    patch_args= ["-p1", "-s"],
    patches = [
        "//third_party/icu:udata.patch",
    ],
)
http_archive(
    name = "re2",
    sha256 = "2ed94072145272012bb5b7054afcbe707447d49dcd79fd6d1689e6f3dc589def",
    strip_prefix = "re2-2019-04-01",
    urls = ["https://github.com/google/re2/archive/2019-04-01.tar.gz"],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

tf_configure(name = "local_config_tf")
cuda_configure(name = "local_config_cuda")
