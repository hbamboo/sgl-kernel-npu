#!/bin/bash
# One-command acceptance run for the ported MlaPrologV3 (GLM-4.7-Flash).
#
#   bash csrc/mla_prolog_v3/tests/run_acceptance.sh [device]
#
# Builds nothing - point it at an already built libsgl_kernel_npu.so and a free device.
set -u
DEV=${1:-5}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT" || exit 1
export ASCEND_RT_VISIBLE_DEVICES="$DEV"
OUT=${OUT:-/tmp/mla_prolog_v3_acceptance}
mkdir -p "$OUT"

pass=0
fail=0
run() {
    local name=$1; shift
    local log="$OUT/$name.log"
    printf '%-34s ' "$name"
    if "$@" > "$log" 2>&1; then
        echo "OK   ($log)"
        pass=$((pass + 1))
    else
        echo "FAIL ($log)"
        tail -3 "$log" | sed 's/^/    /'
        fail=$((fail + 1))
    fi
}

T8=python3
run "golden_T8"            env T=8   $T8 csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py
run "golden_T256"          env T=256 $T8 csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py
run "real_weights_layer2"  $T8 csrc/mla_prolog_v3/tests/test_real_glm_weights.py
run "token_boundary_sweep" $T8 csrc/mla_prolog_v3/tests/sweep_tokens.py
run "vendor_bit_exact"     $T8 csrc/mla_prolog_v3/tests/cmp_builtin_numeric.py
run "out_variant"          $T8 csrc/mla_prolog_v3/tests/test_out_variant.py
run "mxfp8_auto_dequant"   $T8 csrc/mla_prolog_v3/tests/test_sglang_mxfp8_auto.py
run "sglang_alias"         $T8 csrc/mla_prolog_v3/tests/test_sglang_alias.py
run "sglang_nd_weights"    $T8 csrc/mla_prolog_v3/tests/test_sglang_nd_weights.py
run "sglang_mode_infer"    $T8 csrc/mla_prolog_v3/tests/test_sglang_mode_infer.py
run "mlapolog_scheme_state" $T8 csrc/mla_prolog_v3/tests/test_mlapolog_scheme_state.py
run "pybind_fast_path"     $T8 csrc/mla_prolog_v3/tests/test_pybind_path.py
run "perf_sweep"           $T8 csrc/mla_prolog_v3/tests/bench_mla_prolog.py

echo
echo "passed=$pass failed=$fail  (logs in $OUT)"
echo "--- precision summary (golden T=8) ---"
grep -E "cosine=" "$OUT/golden_T8.log" | head -5
echo "--- performance summary ---"
tail -5 "$OUT/perf_sweep.log"
[ "$fail" -eq 0 ]
