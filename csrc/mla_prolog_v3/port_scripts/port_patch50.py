#!/usr/bin/env python3
"""Docs: note the automatic weight_quant_mode inference in the shim."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# module docstring
P1 = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD = """* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively."""
NEW = """* SGLang hardcodes ``weight_quant_mode=1`` in ``forward_mlaprolog``.  The shim derives
  the mode from the actual dtypes (float weights -> 0, fp8 + e8m0 scale -> 3, int8 ->
  keep the caller's 1/2) and drops the scale arguments the derived mode does not use,
  so a dequantised-to-bf16 GLM checkpoint works through the unmodified call site.
* The ported kernel handles ``weight_quant_mode`` 0/1/2 natively."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

# report
P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """   `tests/test_sglang_routing.py` 验证了三件事：标记逻辑在 4 种 ignore 输入下都满足
   SGLang 的谓词、`custom::npu_mla_prolog_v3` 注册成功、`enable_mlaprolog()` 在本镜像
   里确实打上了补丁（`ModelSlimConfig._mlaprolog_patched == True`）。"""
NEW2 = """   `tests/test_sglang_routing.py` 验证了三件事：标记逻辑在 4 种 ignore 输入下都满足
   SGLang 的谓词、`custom::npu_mla_prolog_v3` 注册成功、`enable_mlaprolog()` 在本镜像
   里确实打上了补丁（`ModelSlimConfig._mlaprolog_patched == True`）。

   **第三个坑：SGLang 把 `weight_quant_mode` 硬编码成 1**。GLM 的 MXFP8 权重若已按上文
   反量化成 bf16，再用 mode=1 传入会被 tiling 拒绝。shim 因此在转发前按实际 dtype 推断模式
   （浮点权重→0、fp8+e8m0→3、int8→保留调用方取值），并丢掉该模式用不到的 scale 参数；
   `tests/test_sglang_mode_infer.py` 验证了 4 种 dtype 的推断结果，以及"按 SGLang 原样
   kwargs（mode=1 + bf16 权重 + scale 张量）调用"与直接 mode=0 调用**逐位一致**。

   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）：
   `qkv_a_proj.weight.transpose(0,1)` 后按 `q_lora_rank` 切出 q_a / kv_a（即 [in, out]）、
   `npu_format_cast` 转 NZ、`kv_b_proj.weight.unflatten(0,(-1,D+v)).split([D,v],1)` 取
   `w_kc = [N, D, Hckv]`、`cos_sin_cache[positions]` 后 `chunk(2)` 再 `repeat(1,2)`
   （即 cos/sin 前后半重复）——与本算子的 ABI 完全一致。"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
