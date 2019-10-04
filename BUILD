package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        ".bazelrc",
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "setup.py",
        "//tfmiss",
    ],
)
