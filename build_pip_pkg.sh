#!/usr/bin/env bash
set -e -x

PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"

function main() {
  DEST=${1}
  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  mkdir -p "${DEST}"
  DEST="$(realpath "${DEST}")"
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy module files"
  cp ${PIP_FILE_PREFIX}.bazelrc "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}README.md "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}tfmiss "${TMPDIR}"

  pushd "${TMPDIR}"
  echo $(date) : "=== Building wheel"
  PY_BIN=${PYTHON_BIN_PATH:-`which python`}
  $PY_BIN setup.py bdist_wheel > /dev/null

  # Define OS-specific repair command for
  if [[ ${SKIP_REPAIR:-0} ]]; then
    if which gcp >/dev/null; then
      REPAIR_CMD="gcp -t repaired/"
    else
      REPAIR_CMD="cp -t repaired/"
    fi
  elif [[ $(uname) == "Darwin" ]]; then
    python3 -m pip install -U delocate
    REPAIR_CMD="delocate-wheel -w repaired"
  else
    python3 -m pip install -U auditwheel==2.0.0

    # Patch auditwheel
    AUDIT_WHEEL_PATH=$(python3 -c 'import auditwheel as aw; import os; print(os.path.dirname(aw.__file__))')
    POLICY_JSON="${AUDIT_WHEEL_PATH}/policy/policy.json"
    if [[ ! -f ${POLICY_JSON}.bak ]]; then
      cp -f "${POLICY_JSON}" "${POLICY_JSON}".bak
    fi
    cp -f "${POLICY_JSON}".bak "${POLICY_JSON}"
    TF_SHARED_LIBRARY_NAME=$(grep -r TF_SHARED_LIBRARY_NAME .bazelrc | head -n 1 | awk -F= '{print$2}')
    sed -i "s/libresolv.so.2\"/libresolv.so.2\", ${TF_SHARED_LIBRARY_NAME}/g" ${POLICY_JSON}

    REPAIR_CMD="auditwheel repair --plat manylinux2010_x86_64 -w repaired"
  fi

  # Repair wheels
  mkdir -p repaired
  for WHL in dist/*.whl
  do
    ${REPAIR_CMD} "${WHL}"
  done

  # Move wheels to destination
  cp repaired/*.whl "${DEST}"
  echo $(date) : "=== Output wheel file is in: ${DEST}"

  # Cleanup
  rm -rf repaired/*.whl
  popd
  rm -rf "${TMPDIR}"
}

main "$@"
