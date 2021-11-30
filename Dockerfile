#syntax=docker/dockerfile:1.1.5-experimental
ARG TF_VERSION
ARG PY_VERSION
FROM gcr.io/tensorflow-testing/nosla-cuda11.2-cudnn8.1-ubuntu18.04-manylinux2010-multipython as base_install
ENV TF_NEED_CUDA="1"

# Required for setuptools v50.0.0
# https://setuptools.readthedocs.io/en/latest/history.html#v50-0-0
# https://github.com/pypa/setuptools/issues/2352
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Fix presented in
# https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1/44967506
RUN echo "#! /usr/bin/python2.7" >> /usr/bin/lsb_release2
RUN cat /usr/bin/lsb_release >> /usr/bin/lsb_release2
RUN mv /usr/bin/lsb_release2 /usr/bin/lsb_release

ARG PY_VERSION
RUN ln -sf /usr/local/bin/python$PY_VERSION /usr/bin/python
RUN ln -sf /usr/local/bin/python$PY_VERSION /usr/bin/python3

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ./ /tfmiss
RUN rm /tfmiss/.bazelversion
WORKDIR /tfmiss

# -------------------------------------------------------------------
FROM base_install as make_wheel

RUN python configure.py

#RUN bazel test --test_output=errors //tfmiss/...
RUN bazel build --jobs=4 \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=//third_party/gcc7_manylinux2010-nvcc-cuda11:toolchain build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts

RUN bash auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2010_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

COPY --from=make_wheel /tfmiss/wheelhouse/ /tfmiss/wheelhouse/
RUN pip install /tfmiss/wheelhouse/*.whl

RUN python -c "import tfmiss as tfm;print(tfm.text.zero_digits('123').numpy())"

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /tfmiss/wheelhouse/ .
