"""E2E findings: the kernel now runs inside the real server; the remaining blocker is
SGLang's expectation that `qr` (query_norm) is int8-quantised."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/tests/e2e/README.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## 建议的下一步（按代价排序）"""
NEW = """## 第 18 轮更新：算子已在真实 server 中跑通调用

持续调试后的进展（每一轮都在真实 server 上跑，约 4–5 分钟/次）：

1. **`weight_scale` 缺失** → 给 `NPUFusedMLAPreprocess.__init__` 打补丁补占位 scale；
2. **走错分支** → `forward()` 优先 `forward_mlapo`（int8 路径，需要 `input_offset`），
   补丁改为：投影权重是 fp8 时强制走 `forward_mlaprolog`；
3. **权重方向** → 实测该 checkpoint 下
   * 融合 `qkv_a_proj.weight` 是 **`[1344, 2048]`（`[out, in]`）**，scale_inv `[32, 1344, 2]`
     —— 未转置；
   * `q_b_proj.weight` 是 **`[768, 5120]`（`[in, out]`）**，scale_inv `[12, 5120, 2]`
     —— 已转置；
   两者方向不同，补丁按 scale 的 `out` 维自动判别并反量化；
4. **`_WeightHolder` 必须是 `nn.Module`**，否则无法替换子模块属性；
5. **结果**：`mla_prolog_v3` 在本镜像的 SGLang server 中**调用成功**（不再 tiling 失败），
   收到的三个投影权重均为 bf16 FRACTAL_NZ（2048×768 / 768×5120 / 2048×576）。

### 现在卡在上游的接口假设：`qr` 必须是 int8

调用成功后，SGLang 自己的收尾代码崩了：

```
dequant_q_norm = dequant_q_norm.view(hidden_states.shape[0])
RuntimeError: shape '[1]' is invalid for input of size 0
```

因为 MLAPO 分支（`npu_mla_preprocess`）把算子返回的 `qr` 当**量化后的 q_lora**
（int8 + `dynamic_scale`）交给后续 FIA 量化注意力；而 bf16 权重配置下本算子返回的是
bf16 `query_norm` + 空调度 scale。也就是说：要走通这条路径，`q_b`（以及两个下投影）
必须是 **int8 量化**（`weight_quant_mode=1/2`），而不是 MXFP8/bf16。

## 建议的下一步（按代价排序）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """1. **在 SGLang 侧**给 MLAPO 路径加一个 MXFP8 分支（读取 `weight_scale_inv`、跳过重复转置、
   透传两个下投影的 e8m0 scale），然后即可直接使用原 checkpoint —— 这是最干净的方案。"""
NEW2 = """1. **给 MLAPO 路径加 bf16 支持**（上游改动）：`qr` 以 bf16 返回、`dequant_q_norm` 允许为空，
   后续 FIA 走非量化注意力。这是唯一能保留 MXFP8/bf16 精度的方案。
2. **把三个投影重量化成 int8**（`weight_quant_mode=1`）：SGLang 现有代码可直接工作，但需要
   校准并且精度会低于 MXFP8，与"精度优先"冲突，仅作为兼容选项。
3. **在 SGLang 侧**给 MLAPO 路径加一个 MXFP8 分支（读取 `weight_scale_inv`、跳过重复转置、
   透传两个下投影的 e8m0 scale）。"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("e2e findings updated")
