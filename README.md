# tfmiss
The missing OPs, layer & etc. for TensorFlow

## Development
### Environment
Install all [dependencies including python headers](https://www.tensorflow.org/install/install_sources).
Do not use `pyenv` on MacOS X, otherwise tests mostly likely will fail.

### Build PIP package manually
You can build the pip package with Bazel v0.25.3:
```bash
export PYTHON_BIN_PATH=`which python2.7`
$PYTHON_BIN_PATH -m pip install -U tensorflow  # Only if you did not install it yet
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
