#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Local (sgl-kernel-npu port) fix for CANN's merge_obj_text.sh, which links the
# device object *in place* and is therefore not idempotent: the external-project
# build step runs again on every `cmake --build`, and the second invocation feeds
# an already-merged object back to ld.lld ("unknown file type").
#
# This wrapper keeps the original script untouched and installs the guard.
# ----------------------------------------------------------------------------------------------------------
set -e

UTIL_DIR=/usr/local/Ascend/cann-9.1.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util
ORIG=${UTIL_DIR}/merge_obj_text.sh
BAK=${UTIL_DIR}/merge_obj_text.sh.orig

if [ ! -f "${BAK}" ]; then
    cp -a "${ORIG}" "${BAK}"
    echo "backup written to ${BAK}"
fi

cat > "${ORIG}" <<'EOF'
#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Patched for repeated builds: the merge is skipped when the object has already
# been merged (template text sections renamed to .text), which makes the step
# idempotent.  See merge_obj_text.sh.orig for the original version.
set -e

while [[ $# -gt 0 ]]; do
    case $1 in
    -l | --linker)
        linker=$2
        shift 2
        ;;
    *)
        break
        ;;
    esac
done

for arg in "$@"
do
    if readelf -S "$arg" 2>/dev/null | grep -qE "\.text\._Z|\.text\._ZN|\.text\.mla"; then
        echo -n ""
        ${linker} -m aicorelinux -Ttext=0 ${arg} -o ${arg}
    else
        echo "merge_obj_text: ${arg} already merged, skipping"
    fi
done
EOF
chmod +x "${ORIG}"
echo "patched ${ORIG}"
