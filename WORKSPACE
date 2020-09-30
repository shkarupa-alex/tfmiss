load("//third_party/tf:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

load("//third_party/gpu:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
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
http_archive(
    name = "protobuf_archive",
    sha256 = "1c020fafc84acd235ec81c6aac22d73f23e85a700871466052ff231d69c1b17a",
    strip_prefix = "protobuf-5902e759108d14ee8e6b0b07653dac2f4e70ac73",
    urls = [
        "http://mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/5902e759108d14ee8e6b0b07653dac2f4e70ac73.tar.gz",
    ],
)
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)
http_archive(
    name = "zlib_archive",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "http://mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)
http_archive(
    name = "six_archive",
    build_file = "//third_party:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "http://mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)
bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)
bind(
    name = "six",
    actual = "@six_archive//:six",
)
