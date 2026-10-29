#!/bin/bash
# Set ENABLE_ASCEND_FLASH_MLA_PREFILL=0 and SGLANG_NPU_USE_MLAPO_V3_PREFILL=1
# to keep the prologue enabled with FIA prefill. Decode remains FlashMLA.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh
set -u
export PYTHONPATH="$ROOT/python/sgl_kernel_npu:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7
export SGLANG_NPU_MLA_UNIFIED_KV=1
export ENABLE_ASCEND_FLASH_MLA=1
export ENABLE_ASCEND_FLASH_MLA_PREFILL=${ENABLE_ASCEND_FLASH_MLA_PREFILL:-1}
export SGLANG_NPU_USE_MLAPO_V3_PREFILL=${SGLANG_NPU_USE_MLAPO_V3_PREFILL:-0}
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_NPU_USE_MLAPO_V3=1
export SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE=${SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE:-1}
export SGLANG_NPU_MLAPO_V3_DEBUG=${SGLANG_NPU_MLAPO_V3_DEBUG:-1}
cd /sgl-workspace/sglang
exec python -m sglang.launch_server \
  --model-path /mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8 \
  --host 127.0.0.1 --port "${PORT:-19002}" \
  --device npu --attention-backend ascend --quantization modelslim --tp-size 1 \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  --max-running-requests 92 --cuda-graph-max-bs 92 \
  --context-length 32768 --schedule-policy fcfs --max-prefill-tokens 32768 --chunked-prefill-size 32768 \
  "$@"
