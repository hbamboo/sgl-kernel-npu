#!/usr/bin/env python3
"""Report: document the SGLang drop-in alias and its verification."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """3. **SGLang 接入**：需要在 `mla_preprocess.py` 的 `forward_mlaprolog` 把调用从
   `torch.ops.custom.npu_mla_prolog_v3` 改成 `torch.ops.npu.mla_prolog_v3`（本容器
   `custom_ops` 包没有该算子），并显式打开该分支（当前 checkpoint 的量化描述里没有
   `ignore=kv_b_proj` 白名单）。"""
NEW = """3. **SGLang 接入（已提供免改代码的 shim）**：SGLang 的 `forward_mlaprolog` 调的是
   `torch.ops.custom.npu_mla_prolog_v3`，而本镜像的 `custom_ops` 没有该算子。新增
   `python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py`，在 `custom` 命名空间
   注册同名算子并转发到本移植实现，**签名与 SGLang 的 kwargs 完全一致**
   （token_x / weight_dq / weight_uq_qr / weight_uk / weight_dkv_kr / rmsnorm_gamma_cq /
   rmsnorm_gamma_ckv / rope_sin / rope_cos / kv_cache / kr_cache / cache_index /
   dequant_scale_w_uq_qr / rmsnorm_epsilon_cq / rmsnorm_epsilon_ckv / cache_mode /
   query_norm_flag / weight_quant_mode），返回顺序也一致
   （q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm）：

   ```python
   import sgl_kernel_npu
   from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias
   register_custom_alias()   # 若 custom_ops 已提供则自动跳过，不覆盖官方实现
   ```

   已用 `tests/test_sglang_alias.py` 按 SGLang 的真实 kwargs 验证：解析成功，
   三路输出 `q_nope (T,20,512)` / `q_pe (T,20,64)` / `qr (T,768)` 与直接调用
   **逐位一致**。注意权重需为 NZ 格式，MXFP8 权重需先反量化到 bf16 并置
   `weight_quant_mode=0`。"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("report updated with the SGLang shim")
