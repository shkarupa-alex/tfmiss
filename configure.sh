#!/bin/bash

# Writes variables to bazelrc file
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
  write_to_bazelrc "test --action_env $1=\"$2\""
  write_to_bazelrc ""
}

# Converts the linkflag namespec to the full shared library name
function generate_shared_lib_name() {
  if [[ $(uname) == "Darwin" ]]; then
    local namespec="$1"
    echo "lib${namespec:2}.dylib"
  else
    local namespec="$1"
    echo ${namespec:3}
  fi
}


# Remove .bazelrc if it already exist
[[ -e .bazelrc ]] && rm .bazelrc


# Determine Python path
PYTHON_VERSION=${PYTHON_BIN_PATH:-`which python`}


# Check if TensorFlow installed
if ! ${PYTHON_VERSION} -c "import tensorflow" &> /dev/null; then
    echo 'Install TensorFlow before continuing'
    exit 1
fi


# Store Tensorflow installation path
TF_CFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CXX11_ABI_FLAG=( $(${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)') )

TF_SHARED_LIBRARY_DIR=${TF_LFLAGS[0]:2}
TF_SHARED_LIBRARY_NAME=$(generate_shared_lib_name ${TF_LFLAGS[1]})

write_action_env_to_bazelrc "TF_HEADER_DIR" "${TF_CFLAGS:2}"
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" "${TF_SHARED_LIBRARY_DIR}"
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" "${TF_SHARED_LIBRARY_NAME}"
write_action_env_to_bazelrc "TF_CXX11_ABI_FLAG" "${TF_CXX11_ABI_FLAG}"

# Store common flags
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_to_bazelrc ""
write_to_bazelrc "test --action_env TF_FORCE_GPU_ALLOW_GROWTH=true"
write_to_bazelrc ""

TF_NEED_CUDA=${TF_NEED_CUDA:-"0"}
if [ "${TF_NEED_CUDA:-"0"}" -eq "1" ]; then
  write_action_env_to_bazelrc "TF_NEED_CUDA" "1"
  write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "${CUDA_TOOLKIT_PATH:-"/usr/local/cuda"}"
  write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "${CUDNN_INSTALL_PATH:-"/usr/lib/x86_64-linux-gnu"}"
  write_action_env_to_bazelrc "TF_CUDA_VERSION" "${TF_CUDA_VERSION:-"11"}"
  write_action_env_to_bazelrc "TF_CUDNN_VERSION" "${TF_CUDNN_VERSION:-"8"}"
  write_to_bazelrc ""
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
  write_to_bazelrc ""
  write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
  write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"
fi

