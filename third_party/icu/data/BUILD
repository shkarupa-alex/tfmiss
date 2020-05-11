package(default_visibility = ["//visibility:public"])
licenses(["notice"])  # Apache 2.0
exports_files(["LICENSE"])

# Data for ICU BreakIterator and Normalizer2. Pre-processed for little-endian platforms.
# To replicate this pre-processing (if you want additional encodings, for example),
# do the following:
#
# First, download and unpack ICU.
# Then, run the following from your icu4c/source directory:
#   $ ICU_DATA_FILTER_FILE=<path_to_this_dir>/filter.json ./runConfigureICU Linux
#   $ make clean && make
#   $ LD_LIBRARY_PATH=./lib:./stubdata:./tools/ctestfw:$LD_LIBRARY_PATH ./bin/genccode -f data/out/tmp/icudt64l data/out/tmp/icudt64l.dat
#   $ echo "U_CAPI const void * U_EXPORT2 uprv_getICUData_other() { return icudt64l_dat.bytes; }" >> data/out/tmp/icudt64l.c
#   $ mv data/out/tmp/icudt64l.c data/out/tmp/icudt.c
#   $ gzip data/out/tmp/icudt.c
#   $ split -b 176k data/out/tmp/icudt.c.gz icudt.c.gz.
#   $ mv icudt.c.gz.* <path_to_this_dir>/

filegroup(
    name = "icudt_files",
    srcs = glob(["icudt.c.gz.*"]),
)

# Data files are compressed and split to work around git performance degradation around large files.
genrule(
    name = "merge_icudt",
    srcs = [":icudt_files"],
    outs = ["icudt.c"],
    cmd = "cat $(locations :icudt_files) | gunzip > $@",
)

cc_library(
    name = "custom_data",
    srcs = [":icudt.c"],
    deps = ["@icu//:headers"],
    alwayslink = 1,
)
