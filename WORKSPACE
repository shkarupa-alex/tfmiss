load("//third_party/tf:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

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
    build_file = "//third_party/icu:BUILD.bazel",
    patch_args= ["-p1"],
    patches = [
        "//third_party/icu:udata.patch",
    ],
)
http_archive(
    name = "re2",
    sha256 = "88864d7f5126bb17daa1aa8f41b05599aa6e3222e7b28a90e372db53c1c49aeb",
    strip_prefix = "re2-2020-05-01",
    urls = [
        "https://mirror.bazel.build/github.com/google/re2/archive/2020-05-01.tar.gz",
        "https://github.com/google/re2/archive/2020-05-01.tar.gz"
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
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"],
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
    sha256 = "d16a0141ec1a18405cd4ce8b4613101da75da0e9a7aec5bdd4fa804d0e0eba73",
    strip_prefix = "six-1.12.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
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