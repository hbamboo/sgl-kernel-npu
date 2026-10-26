# SGLang 端到端验证记录（GLM-4.7-Flash + MlaPrologV3）

本目录是第 15–17 轮做"模型级端到端"验证时用的脚本与结论。

## 文件

| 文件 | 用途 |
|---|---|
| `sitecustomize.py` | 通过 `PYTHONPATH` 在解释器启动时打补丁（注册 `custom::npu_mla_prolog_v3` 别名 + 打开 MLAPO 路由） |
| `run_server.sh` | 用 MLAPO 路径（本算子）起 SGLang server |
| `run_server_baseline.sh` | 同参数但 `SGLANG_NPU_USE_MLAPO=0`，作为对照 |

启动方式：

```bash
cd csrc/mla_prolog_v3/tests/e2e
setsid nohup bash run_server.sh > /tmp/e2e.log 2>&1 < /dev/null &
# 约 3–4 分钟（31 GB 权重），日志出现 "The server is fired up" 即可
curl -s http://127.0.0.1:30111/generate -H "Content-Type: application/json" \
     -d '{"text":"用一句话解释什么是 MLA：","sampling_params":{"temperature":0,"max_new_tokens":48}}'
```

## 结论（累计证据）

1. **补丁链路是通的**：`[dsh-patch] custom alias: True` / `mlaprolog routing: True`，
   且 SGLang 确实构造了 `NPUFusedMLAPreprocess`（第 15 轮 traceback 与第 17 轮成功启动都证明了这一点）。
2. **对照基线可用**：原始 checkpoint + `SGLANG_NPU_USE_MLAPO=0` + `--quantization modelslim`
   在 device 5 上正常出词：

   > MLA 是一种通过压缩键值（KV）缓存来降低显存占用、提升推理效率的模型架构创新。…

3. **MXFP8 checkpoint 无法直接走 MLAPO**：SGLang 的 `forward_mlaprolog` 假设 W8A8-int8
   modelslim scheme，而 MXFP8 scheme 会
   * 删除 `weight_scale`（改名为 `weight_scale_inv`），→ 构造函数 `AttributeError`；
   * 把 `weight` 预先转置成 `[in, out]`，→ `mlaprolog_preprocess_weight` 里的
     `transpose(0,1)` 又转回去，切片结果错位；
   * `forward_mlaprolog` 只透传 `dequant_scale_w_uq_qr`，两个下投影的 scale 根本没传进来。
   这三条是**上游 SGLang 与 MXFP8 scheme 的不兼容**，与本算子无关。
4. **bf16 overlay 方案（`tools/convert_mla_proj_to_bf16.py`）尚未打通**：
   把三个投影反量化成 bf16 并标为 FLOAT 后，模型输出仍然是乱码（且与 MLAPO 开关无关），
   说明加载器对"BF16 的融合 qkv_a（q_a/kv_a 合并）"的处理与我构造的 overlay 不一致，
   需要继续排查 SGLang 的权重合并/加载路径。

## 第 18 轮更新：算子已在真实 server 中跑通调用

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

## 建议的下一步（按代价排序）

1. **给 MLAPO 路径加 bf16 支持**（上游改动）：`qr` 以 bf16 返回、`dequant_q_norm` 允许为空，
   后续 FIA 走非量化注意力。这是唯一能保留 MXFP8/bf16 精度的方案。
2. **把三个投影重量化成 int8**（`weight_quant_mode=1`）：SGLang 现有代码可直接工作，但需要
   校准并且精度会低于 MXFP8，与"精度优先"冲突，仅作为兼容选项。
3. **在 SGLang 侧**给 MLAPO 路径加一个 MXFP8 分支（读取 `weight_scale_inv`、跳过重复转置、
   透传两个下投影的 e8m0 scale）。
2. 继续调试 bf16 overlay：对比 `fused_qkv_a_proj_with_mqa.weight` 在加载后与
   `cat(dequant(q_a), dequant(kv_a))` 是否一致，定位合并环节。
3. 包一层 `forward_mlaprolog`：在调用前把三个权重换成 bf16 NZ、并显式传入 scale，
   完全绕开 SGLang 的权重预处理（工作量中等，但不动模型加载器）。
