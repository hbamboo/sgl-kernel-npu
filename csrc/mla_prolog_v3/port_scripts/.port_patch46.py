#!/usr/bin/env python3
"""Report: document the two-step SGLang integration (alias + routing patch)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """   已用 `tests/test_sglang_alias.py` 按 SGLang 的真实 kwargs 验证：解析成功，
   三路输出 `q_nope (T,20,512)` / `q_pe (T,20,64)` / `qr (T,768)` 与直接调用
   **逐位一致**。注意权重需为 NZ 格式，MXFP8 权重需先反量化到 bf16 并置
   `weight_quant_mode=0`。"""
NEW = """   已用 `tests/test_sglang_alias.py` 按 SGLang 的真实 kwargs 验证：解析成功，
   三路输出 `q_nope (T,20,512)` / `q_pe (T,20,64)` / `qr (T,768)` 与直接调用
   **逐位一致**。注意权重需为 NZ 格式，MXFP8 权重需先反量化到 bf16 并置
   `weight_quant_mode=0`。

   **另外还有一个必踩的坑**：`mla_preprocess.py` 用
   `any(re.fullmatch(r".*kv_b_proj", l) for l in quant_config.ignore)` 决定是否走
   `forward_mlaprolog`，而 modelslim 的 `ModelSlimConfig` 是从 quant 描述里的
   `"ignore"` 键读取该列表——本 checkpoint 的 `quant_model_description.json`
   **没有这个键**（它是扁平的 权重名→dtype 映射，`kv_b_proj.weight` 标为 FLOAT），
   于是 `ignore == []`，**即使装了 custom_ops 也永远不会走 MLAPO 分支**。
   为此 compat 模块另外提供：

   ```python
   from sgl_kernel_npu.mla_prolog_v3_compat import enable_mlaprolog
   enable_mlaprolog()   # patch ModelSlimConfig.__init__，补上 kv_b_proj 标记
   ```

   `tests/test_sglang_routing.py` 验证了三件事：标记逻辑在 4 种 ignore 输入下都满足
   SGLang 的谓词、`custom::npu_mla_prolog_v3` 注册成功、`enable_mlaprolog()` 在本镜像
   里确实打上了补丁（`ModelSlimConfig._mlaprolog_patched == True`）。"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("report updated")
