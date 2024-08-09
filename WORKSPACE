load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/gpu:cuda_configure.bzl", "cuda_configure")
load("//third_party/tf:tf_configure.bzl", "tf_configure")

# curl -L https://github.com/.../.../archive/<git hash>.tar.gz | [g]sha256sum

# See ICU data redbuild instructions in third_party/icu/data/BUILD
#http_archive(
#    name = "icu",
#    sha256 = "925e6b4b8cf8856e0ac214f6f34e30dee63b7bb7a50460ab4603950eff48f89e",
#    strip_prefix = "icu-release-75-1",
#    urls = ["https://github.com/unicode-org/icu/archive/release-75-1.tar.gz"],
#    build_file = "//third_party/icu:BUILD.bzl",
#    patch_args= ["-p1", "-s"],
#    patches = [
#        "//third_party/icu:udata.patch",
#    ],
#)
#http_archive(
#    name = "icu",
#    sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
#    strip_prefix = "icu-release-65-1",
#    urls = ["https://github.com/unicode-org/icu/archive/release-65-1.tar.gz"],
#    build_file = "//third_party/icu:BUILD.bzl",
#    patch_args= ["-p1", "-s"],
#    patches = [
#        "//third_party/icu:udata.patch",
#    ],
#)
http_archive(
    name = "icu",
    build_file = "//third_party/icu:BUILD.bzl",
    patch_args = [
        "-p1",
        "-s",
    ],
    patches = [
        "//third_party/icu:udata.patch",
    ],
    sha256 = "65271a83fa81783d1272553f4564965ac2e32535a58b0b8141e9f4003afb0e3a",
    strip_prefix = "icu-release-64-2",
    urls = ["https://github.com/unicode-org/icu/archive/release-64-2.tar.gz"],
)

http_archive(
    name = "re2",
    sha256 = "2ed94072145272012bb5b7054afcbe707447d49dcd79fd6d1689e6f3dc589def",
    strip_prefix = "re2-2019-04-01",
    urls = ["https://github.com/google/re2/archive/2019-04-01.tar.gz"],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")
