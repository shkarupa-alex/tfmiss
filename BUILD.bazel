package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "requirements.txt",
        "setup.py",
        "//tfmiss",
    ],
)
