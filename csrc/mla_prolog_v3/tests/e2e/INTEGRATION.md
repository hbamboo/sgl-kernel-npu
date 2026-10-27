# 在自有服务里接入移植后的 MlaPrologV3（SGLang / NPU）

本文面向"想在自己的 SGLang 启动脚本里用上这个算子"的读者。**不需要改 SGLang 源码**，
但需要知道两件事：怎么挂补丁，以及当前 checkpoint 的量化形态会走到哪条上游分支。

---

## 1. 三个前提

| # | 前提 | 说明 |
|---|---|---|
| 1 | 权重形态 | 三个投影必须是 **bf16 + FRACTAL_NZ**（PyBind 路径要求），或**fp8 + e8m0 scale**（高层 API / torch.ops shim 会自动反量化）。`weight_uk`（`w_kc`）保持 bf16 ND。 |
| 2 | 权重朝向 | 两种都接受：checkpoint 原样的 `[out, in]`（按 scale 自动判别并转置），或算子 ABI 的 `[in, out]`。 |
| 3 | 模型量化形态 | SGLang 的 MLAPO 融合链路（`forward_mlapolog` / `forward_mlapo`）**围绕 W8A8 int8 设计**。MXFP8 checkpoint 会在 prefill 阶段走进 int8 分支并因缺 `input_offset` 报错——这是上游契约，不是算子问题（详见第 5 节）。 |

---

## 2. 挂补丁（30 秒）

在**模型加载之前**执行：

```python
import sgl_kernel_npu                      # 载入 libsgl_kernel_npu.so（内含 torch.ops.npu.* 与 PyBind 绑定）
from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias, enable_mlaprolog

register_custom_alias()   # 让 torch.ops.custom.npu_mla_prolog_v3 解析到本实现（若 custom_ops 已提供则自动跳过）
enable_mlaprolog()        # 让 SGLang 的 _is_mlaprolog 命中（补 kv_b_proj 标记）
```

推荐用 `sitecustomize.py` 自动执行（本目录已有现成文件）：

```bash
export PYTHONPATH=<repo>/csrc/mla_prolog_v3/tests/e2e:<repo>/python/sgl_kernel_npu:$PYTHONPATH
export DSH_MLA_PROLOG_V3_PATCH=1     # sitecustomize 里用它做开关
export SGLANG_NPU_USE_MLAPO=1        # SGLang 自己的 MLAPO 总开关
```

启动示例见本目录 `run_server.sh`（含 `--quantization modelslim`、`--attention-backend ascend` 等）。

---

## 3. 不启 server 的最小验证

```bash
python3 csrc/mla_prolog_v3/tests/test_sglang_alias.py       # 别名解析 + 与直接调用逐位一致
python3 csrc/mla_prolog_v3/tests/test_sglang_mode_infer.py   # weight_quant_mode 自动推断
python3 csrc/mla_prolog_v3/tests/test_sglang_nd_weights.py   # ND 权重自动转 NZ
python3 csrc/mla_prolog_v3/tests/test_shim_ckpt_orientation.py  # checkpoint 朝向（[out,in]）也能吃
```

四个都通过，说明"torch.ops 侧接线"没问题，可以再启 server。

---

## 4. 直接用算子（不经过 SGLang）

```python
from sgl_kernel_npu.mla_prolog_v3 import mla_prolog_v3     # 高层 API：自动反量化/转向/NZ/模式推断

q, q_rope, dq_nope, q_norm, dq_norm = mla_prolog_v3(
    x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, kv_cache, kr_cache,
    cache_index=slot_mapping,
    dequant_scale_w_dq=s_dq, dequant_scale_w_uq_qr=s_uq, dequant_scale_w_dkv_kr=s_dkv,
    query_norm_flag=True)
```

服务循环里建议改用 `mla_prolog_v3_out`（自带缓冲区）或 PyBind 直调（见 `mla_prolog_v3_compat.direct()`），
解码态分别快 ~13% 与 ~21%。

---

## 5. 当前 MXFP8 checkpoint 的边界（重要）

实测结论（`tests/e2e/README.md` 有完整日志）：

| 现象 | 根因 |
|---|---|
| `ReplicatedLinear has no attribute 'input_offset'` | prefill 走 `forward_mlapo`（int8 融合路径），MXFP8 没有 W8A8 的 `input_offset` |
| `ColumnParallelLinear has no attribute 'weight_scale'` | MXFP8 scheme 把 `weight_scale` 改名 `weight_scale_inv` 并删除 |
| `too many values to unpack (expected 7)` | 是否走 MLAPO 由调用方按 `quant_config.ignore` 决定，强行改道会破坏解包契约 |

**结论**：MLAPO 融合链路要支持 MXFP8，需要在 SGLang 侧新增分支（读 `weight_scale_inv`、
跳过重复转置、prefill 走非 int8 路径）；在算子侧无法单方面解决。

---

## 6. 故障排查表

| 症状 | 检查 |
|---|---|
| `mla_prolog_v3 tiling failed` | 设 `DSH_MLA_V3_DEBUG=1` 重跑，会打印每个张量的形状/类型/属性；最常见是 `query_norm` 宽度写成 `He` 而非 `Hcq`，或空输出被写成多维零张量 |
| 结果全是噪声 | 权重不是 NZ（ND 会得到 cosine≈0.005）；改用 `ensure_nz` 或换成高层 API |
| SGLang 仍走旧路径 | `SGLANG_NPU_USE_MLAPO=1` 是否设置；`enable_mlaprolog()` 是否在模型加载前调用 |
| 找不到 `custom::npu_mla_prolog_v3` | `register_custom_alias()` 是否调用；`PYTHONPATH` 是否包含 `python/sgl_kernel_npu` |
| 精度对不上 golden | 先跑 `run_acceptance.sh`（20 项）；若只有某几层错，检查该层 checkpoint 的 scale 是否存在 |
