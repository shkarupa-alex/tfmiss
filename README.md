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
export TF_CUDA_VERSION="11.8"
export TF_CUDNN_VERSION="8"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

./configure.py
bazel clean --expunge
bazel test --test_output=errors //tfmiss/...
bazel build build_pip_pkg
bazel-bin/build_pip_pkg wheels
```

### Build release with Linux docker container
```bash
# Requires about 4Gb of RAM allocated to Docker
DOCKER_BUILDKIT=1 docker build -t miss --output type=local,dest=wheels --build-arg PY_VERSION=3.8 ./
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
