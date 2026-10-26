#!/usr/bin/env python3
"""Docs: record the MXFP8 auto-dequant in the shim."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P1 = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD = """* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively.  For the MXFP8 GLM
  checkpoint (mode 3) the ``weight_uq_qr`` per-32 scale mapping is still incorrect for
  ``Hcq=768``, so dequantise the MXFP8 weights to bf16 offline and use mode 0 (see
  ``csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md``)."""
NEW = """* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively; its native MXFP8 path
  (mode 3) mis-maps the ``weight_uq_qr`` per-32 scales at ``Hcq=768``.  The shim therefore
  dequantises float8_e4m3 weights + e8m0 scales to bf16 on first use (cached, both the
  op's ``[in, out]`` and the checkpoint's ``[out, in]`` scale orientations) and switches
  the call to mode 0, so the MXFP8 checkpoint can be passed through unchanged.
  ``clear_dequant_cache()`` drops the cache."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）："""
NEW2 = """   **第五个坑：MXFP8 权重**。kernel 原生 MXFP8 路径在 `Hcq=768` 下 scale 映射有误，唯一
   数值正确的配置是 bf16 + mode 0。shim 因此在首次调用时把 `float8_e4m3` 权重按 e8m0 尺度
   反量化成 bf16 并缓存（同时支持算子 ABI 的 `[in,out]` 与 checkpoint 的 `[out,in]` 两种
   scale 朝向），随后把调用切到 mode 0。`tests/test_sglang_mxfp8_auto.py` 用 **真实 layer-2
   的 fp8 权重 + e8m0 scale**（按 SGLang 的转置/切分方式组织）直接喂给 shim，五路输出
   对 CPU golden 的 cosine 为 0.9999976 / 0.9999979 / 0.9999979 / 0.9999979 / 0.9999982。

   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）："""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
