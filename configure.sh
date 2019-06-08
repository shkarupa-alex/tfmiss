#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
  write_to_bazelrc "test --action_env $1=\"$2\""
  write_to_bazelrc ""
}


# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc


# Determine Python path
PY_BIN=${PYTHON_BIN_PATH:-`which python`}


# Check if TensorFlow installed
if ! $PY_BIN -c "import tensorflow" &> /dev/null; then
    echo 'Install TensorFlow before continuing'
    exit 1
fi


# Store Tensorflow installation path
TF_CFLAGS=( $($PY_BIN -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $($PY_BIN -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
if [ -f "${SHARED_LIBRARY_DIR}/libtensorflow_framework.dylib" ]; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
fi
if [ ! -f "${SHARED_LIBRARY_DIR}/${SHARED_LIBRARY_NAME}" ]; then
    echo "TensorFlow shared library not found in ${SHARED_LIBRARY_DIR}"
    exit 1
fi
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}


# Store old GLIB compatibility flags
TF_CXX11_ABI_FLAG=( $($PY_BIN -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" ${TF_CXX11_ABI_FLAG}


# Store common flags
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_to_bazelrc ""
write_to_bazelrc "test --spawn_strategy=standalone"
write_to_bazelrc "test --strategy=Genrule=standalone"
