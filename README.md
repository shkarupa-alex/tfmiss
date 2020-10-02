# tfmiss
The missing OPs, layer & etc. for TensorFlow

## Development
### Environment
Install all [dependencies including python headers](https://www.tensorflow.org/install/install_sources).
Do not use `pyenv` on MacOS X, otherwise tests mostly likely will fail.

### Build PIP package manually
You can build the pip package with Bazel v0.25.3:
```bash
# GPU support
export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="10.1"
export TF_CUDNN_VERSION="7"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# Set this to proper GCC v8 compiller path if using GPU
export GCC_HOST_COMPILER_PATH=`which gcc-8`

# Set these to target another python interpreter
export PYTHON_BIN_PATH=`which python`

./configure.sh
bazel clean --expunge
bazel test --test_output=errors //tfmiss/...
bazel build build_pip_pkg
bazel-bin/build_pip_pkg wheels
```

### Build release with Linux docker container
```bash
docker run -it -v `pwd`:/tfmiss tensorflow/tensorflow:custom-op-ubuntu16 /tfmiss/build_linux_release.sh
```

### Install and test PIP package
Once the pip package has been built, you can install it with:
```bash
pip install wheels/*.whl
```

Now you can test out the pip package:
```bash
cd /
python -c "import tensorflow as tf;import tfmiss as tfm;print(tfm.text.zero_digits('123').numpy())"
```

You should see the op zeroed out all nonzero digits in string "123":
```bash
000
```
