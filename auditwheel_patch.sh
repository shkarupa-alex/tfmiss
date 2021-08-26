set -e -x

SITE_PKG_LOCATION=$(python -c "import site; print(site.getsitepackages()[0])")
TF_SHARED_LIBRARY_NAME=$(grep -r TF_SHARED_LIBRARY_NAME .bazelrc | awk -F= '{print$2}')
POLICY_JSON="${SITE_PKG_LOCATION}/auditwheel/policy/policy.json"
sed -i "s/libresolv.so.2\"/libresolv.so.2\", $TF_SHARED_LIBRARY_NAME/g" $POLICY_JSON