load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/gpu:cuda_configure.bzl", "cuda_configure")
load("//third_party/tf:tf_configure.bzl", "tf_configure")

# See ICU data redbuild instructions in third_party/icu/data/BUILD
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
    strip_prefix = "icu-release-64-2",
    urls = ["https://github.com/unicode-org/icu/archive/release-64-2.tar.gz"],
)

http_archive(
    name = "com_google_absl",
    # Match commit with https://github.com/tensorflow/tensorflow/blob/v2.17.0/third_party/absl/workspace.bzl#L10
    strip_prefix = "abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30",
    urls = ["https://github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz"],
)

http_archive(
    name = "re2",
    strip_prefix = "re2-2023-09-01",
    urls = ["https://github.com/google/re2/archive/2023-09-01.tar.gz"],
)

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

tf_configure(name = "local_config_tf")

cuda_configure(name = "local_config_cuda")
