"""E2E: record the round-19 findings and the final scope statement."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/tests/e2e/README.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## 建议的下一步（按代价排序）"""
NEW = """## 第 19 轮更新：为什么"强行改道"不可行

试过给 `dequant_q_norm` 返回合法占位 scale（`ones(T)`），以及把 `forward()` 强制改道到
`forward_mlaprolog`。结果暴露出更深的一层约束：

```
ValueError: too many values to unpack (expected 7)
```

调用方 `npu_mla_preprocess` 自己按 `quant_config.ignore` 判断该解包 7 个还是 8 个返回值；
而本模型在该分支里**没有被判定为 mlaprolog**（`quant_config` 没有 `ignore` 标记），于是它走
else 分支解包 7 个，我们却返回了 8 个。也就是说：**是否走 MLAPO 是调用方的决定，算子无法
单方面接管**（强行接管只会破坏解包契约）。该改道逻辑已移除。

### 结论（本轮定稿）

* **算子本身已在真实 server 中被成功调用**（权重 bf16 NZ、无 tiling 错误，已见第 18 轮日志）；
* SGLang 这条路径的**契约**是：三个投影 W8A8 int8 + `kv_b_proj` 未量化 + `ignore` 标记，
  并由调用方选择 MLAPO 分支；本移植算子在该配置下原生支持（`weight_quant_mode=1`）；
* 对**本 MXFP8 checkpoint**，SGLang 侧既不选 MLAPO 分支、也不接受 bf16 `qr`，属于上游
  功能缺口，需要在 SGLang 里做（明确的三条路线见下）。

## 建议的下一步（按代价排序）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("e2e readme updated")
