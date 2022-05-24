#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
ARG PY_VERSION
FROM tensorflow/build:latest-python$PY_VERSION as base_install

ENV TF_NEED_CUDA="1"
ENV DOCKER_BUILD="1"

# TODO: Remove this if tensorflow/build container removes their keras-nightly install
# https://github.com/tensorflow/build/issues/78
RUN python -m pip uninstall -y keras-nightly

COPY ./ /tfmiss
WORKDIR /tfmiss

RUN python -m pip install -r requirements.txt

# -------------------------------------------------------------------
FROM base_install as make_wheel

RUN python configure.py

#RUN bazel test --test_output=errors //tfmiss/...
RUN bazel build --jobs=4 \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=//third_party/gcc9_manylinux2014-nvcc-cuda11:toolchain build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts

RUN bash auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------
FROM python:$PY_VERSION as test_wheel_in_fresh_environment

COPY --from=make_wheel /tfmiss/wheelhouse/ /tfmiss/wheelhouse/
RUN pip install /tfmiss/wheelhouse/*.whl

RUN python -c "import tfmiss as tfm;print(tfm.text.zero_digits('123').numpy())"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /tfmiss/wheelhouse/ .
