#!/usr/bin/env python3
"""Docs: record the NZ auto-cast and the three SGLang integration pitfalls."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P1 = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD = """* Weights must be NZ (``torch_npu.npu_format_cast(w, 29)``) for the three projection
  matrices, matching what ``mlaprolog_preprocess_weight`` produces."""
NEW = """* The kernel needs FRACTAL_NZ for the three projection matrices.  SGLang only NZ-casts
  the fused qkv_a projection and hands over ``q_b_proj.weight`` / ``w_kc`` in ND (the
  MXFP8 scheme keeps them ND on purpose), and ND weights make the kernel produce
  garbage (cosine 0.005).  The shim therefore casts ND -> NZ on the fly and caches the
  result (weights are static after loading; ``clear_format_cache()`` drops it)."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）："""
NEW2 = """   **第四个坑（最隐蔽）：`q_b_proj` 是 ND 格式**。`mlaprolog_preprocess_weight` 只对
   融合后的 `qkv_a_proj` 做 `npu_format_cast`（→NZ），`q_b_proj.weight` 与 `w_kc` 原样
   传入（MXFP8 scheme 的 `process_weights_after_loading` 明确走"只转置、不转 NZ"分支）。
   而本 kernel 的三个投影矩阵必须是 NZ（ND 会算出 cosine 0.005 的垃圾值）。shim 因此
   在转发前对 `weight_dq/weight_uq_qr/weight_dkv_kr` 做 ND→NZ 转换并缓存（权重加载后
   静态，`clear_format_cache()` 可清缓存）。`tests/test_sglang_nd_weights.py` 模拟真实
   情形（qkv_a 为 NZ、q_b 为 ND、`weight_quant_mode=1`），结果与"全 NZ + mode 0"
   参考调用**逐位一致**，且第二次调用命中缓存后仍一致。

   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）："""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
