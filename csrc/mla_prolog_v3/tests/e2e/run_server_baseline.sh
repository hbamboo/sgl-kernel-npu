#!/bin/bash
# Baseline run: same checkpoint, MLAPO path disabled, so the ported kernel is NOT used.
set -u
export PYTHONPATH="/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/tests/e2e:/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu:${PYTHONPATH:-}"
export SGLANG_NPU_USE_MLAPO=0
export ASCEND_USE_FIA=1
export ENABLE_ASCEND_FLASH_ATTN=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-5}
PORT=${PORT:-30112}
MODEL=${MODEL:-/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8-mlaprolog-bf16}

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --device npu --attention-backend ascend \
  --quantization modelslim --tp-size 1 \
  --mem-fraction-static 0.55 \
  --max-running-requests 4 --cuda-graph-max-bs 4 \
  --disable-cuda-graph \
  --trust-remote-code
