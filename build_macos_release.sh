#!/usr/bin/env bash
set -e -x

PYTHON_VERSIONS="2.7.16 3.6.9 3.7.4"
curl -sSOL https://bootstrap.pypa.io/get-pip.py

# Install Bazel 0.24
# wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-darwin-x86_64.sh
# chmod +x bazel-0.24.1-installer-darwin-x86_64.sh
# ./bazel-0.24.1-installer-darwin-x86_64.sh --user
# export PATH="$PATH:$HOME/bin"

# brew update && (brew upgrade pyenv || brew install pyenv)
# eval "$(pyenv init -)"

for version in ${PYTHON_VERSIONS}; do
    export PYENV_VERSION=${version}
    pyenv install -s "${PYENV_VERSION}"

    python get-pip.py -q
    python -m pip --version
    python -m pip install -U -q delocate

    # Link TF dependency
    ./configure.sh

    # Test & build package
    bazel clean  # --expunge
    bazel test --test_output=errors //tfmiss/...
    bazel build build_pip_pkg

    # Package Whl
    bazel-bin/build_pip_pkg wheels
done

# Clean up
rm get-pip.py
