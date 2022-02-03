load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/tf:tf_configure.bzl", "tf_configure")
load("//third_party/gpu:cuda_configure.bzl", "cuda_configure")

# curl -L https://github.com/.../.../archive/<git hash>.tar.gz | [g]sha256sum

http_archive(
    name = "icu",
    sha256 = "524960ac99d086cdb6988d2a92fc163436fd3c6ec0a84c475c6382fbf989be05",
    strip_prefix = "icu-release-64-2",
    urls = [
        "https://mirror.bazel.build/github.com/unicode-org/icu/archive/release-64-2.tar.gz",
        "https://github.com/unicode-org/icu/archive/release-64-2.tar.gz",
    ],
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
    urls = [
        "https://mirror.bazel.build/github.com/google/re2/archive/2019-04-01.tar.gz",
        "https://github.com/google/re2/archive/2019-04-01.tar.gz"
    ],
)

tf_configure(name = "local_config_tf")
cuda_configure(name = "local_config_cuda")
