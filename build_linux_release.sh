#!/usr/bin/env bash
set -e -x

PYTHON_VERSIONS="python2.7 python3.6 python3.7"
ln -sf /usr/bin/python3.5 /usr/bin/python3 # Py36 has issues with add-apt

rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
add-apt-repository -y ppa:deadsnakes/ppa
apt-get -y -q update

cp -R /tfmiss ~/tfmiss && cd ~/tfmiss
mkdir -p /tfmiss/wheels
curl -sSOL https://bootstrap.pypa.io/get-pip.py

for VERSION in ${PYTHON_VERSIONS}; do
    export PYTHON_VERSION=${VERSION}
    apt-get -y -q install "${PYTHON_VERSION}"

    # Required env var for building package
    export PYTHON_BIN_PATH=`which ${PYTHON_VERSION}`

    # Update pip
    ${PYTHON_BIN_PATH} get-pip.py -q

    # Install TF & dependences
    ${PYTHON_BIN_PATH} -m pip install -U setuptools pandas numpy matplotlib tensorflow==2.0.0

    # Link TF dependency
    ./configure.sh

    # Test & build package
    bazel clean  # --expunge
    #bazel test \
    #  --crosstool_top=//third_party/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
    #  --test_output=errors \
    #  //tfmiss/...
    bazel build \
      --crosstool_top=//third_party/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
      build_pip_pkg

    # Build wheel
    bazel-bin/build_pip_pkg /tfmiss/wheels
done
