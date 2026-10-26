#!/usr/bin/env python3
"""e2e README: record why MLAPO cannot serve an MXFP8 checkpoint end to end."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/tests/e2e/README.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## 建议的下一步（按代价排序）"""
NEW = """## 第 35 轮更新：强制 MLAPO 也走不通——**prefill 阶段用的是 int8 路径**

试过用 `DSH_MLA_PROLOG_V3_FORCE_MLAPO=1` 把
`deepseek_v2_attention_mla_npu.npu_mla_preprocess` 整体替换成恒走
`forward_mlaprolog`（8 返回值）的版本。结果：

* decode 阶段确实进了本算子（`[dsh-patch] forced the MLAPO path` 打印成功）；
* 但 **prefill 阶段**报 `ReplicatedLinear has no attribute 'input_offset'` —— 因为
  `NPUFusedMLAPreprocess.forward()` 的分派是
  `if _is_w8a8: forward_mlapo(...)`（int8 融合路径，需要 W8A8 的 `input_offset`），
  而 modelslim MXFP8 也满足 `_is_w8a8` 的判定（`quant_method.quantization_config`
  的 name 同样是 "modelslim"）。

也就是说：**SGLang 的 MLAPO 相关融合算子整体是围绕 W8A8 int8 设计的**，
MXFP8 checkpoint 不仅缺 `weight_scale`/`input_offset`，连 prefill 都会走进 int8 分支。
这条链路要支持 MXFP8，必须在 SGLang 侧新增分支，而不是在调用处包装。

（该实验补丁已回滚，未留在代码里。）

## 建议的下一步（按代价排序）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("e2e readme updated")
