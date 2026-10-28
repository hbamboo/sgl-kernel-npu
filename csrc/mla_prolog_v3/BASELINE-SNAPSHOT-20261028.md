# 基线固化快照（2026-10-28，第 5 轮）

> 本轮第一件事是把已达成并验证过的结果固化成可追溯的基线，然后再推进 1 ms。
> 二进制被 `.gitignore` 的 `*.so` 排除，所以这里记录 md5 与构建/验收方式，
> 保证"某个 commit ↔ 某个 .so ↔ 某组实测数字"三者可对应。

## 1. 基线身份

| 项 | 值 |
|---|---|
| commit | `3ae0be3`（docs: the step ceiling is L1_A plus fixed L0 offsets, not the vector loop）|
| 工作树 | 干净 |
| `.so` md5 | `439943cbbe88352b22936f16d09b0eb1` |
| `.so` 大小 / 时间 | 3,080,480 B / 2026-10-28 11:02 |
| 构建 | `cmake --build /tmp/bk --target sgl_kernel_npu -j 16`，CANN 9.1.0，`SOC_VERSION=Ascend950PR_9599` |
| 设备 | Ascend950DT（本轮实测在 device 0；device 5/6/7 上数字一致）|

## 2. 性能基线（本快照实测）

| 场景 | 数值 |
|---|---|
| prefill T=30456，kernel（`torch.ops.npu.mla_prolog_v3`，MXFP8）| **2336 µs**（225.6 TFLOP/s）|
| prefill T=30456，Python API + 真实 checkpoint 权重 | **2230 µs**（237 TFLOP/s）|
| prefill T=8192，MXFP8 | 678 µs |
| prefill T=8192，bf16 | 2602 µs |
| decode T=1 单次 | **41.4 µs** |
| decode 47 层（一个 step）| **1095 µs** |

## 3. 精度基线

| 项 | 数值 |
|---|---|
| MXFP8 合成几何（`test_mxfp8_precision.py`，T=8）| query 0.9995467 / qrope 0.9995407 / kv 0.9999989 / kr 0.9999987 |
| 真实 checkpoint layer 2（五路）| 0.9999975 / 0.9999979 / 0.9999979 / 0.9999980 / 0.9999982 |
| split-M vs split-N | **逐位相同**（T=8192 / 30456 五路 max\|d\| = 0）|
| 验收 | **27/27** |

## 4. 这个基线实现了什么

| 目标 | 状态 |
|---|---|
| ① MXFP8 可用、精度正常、有性能收益 | ✅ 0.9761→0.99956；9611 µs→5044 µs（vs bf16），再叠加 split-M → 2336 µs |
| ② prefill 30456 < 2.5 ms | ✅ 2336 µs（kernel）/ 2230 µs（API），低 6.5%–10.8% |
| ③ 1 ms | ❌ 见 §5 |

## 5. 已被实验否定的路径（不要重复）

| 假设 | 结果 | 轮次 |
|---|---|---|
| HBM 是瓶颈（324 MB/pass × 8）| 错。L2 命中 95.9%，真实瓶颈是每核 L1 填充 6.7 GB/s | 3 |
| 减小核数省带宽 | 错。8/16/24 核 → 7916/4177/2927 µs，32 核最优 | 1、3 |
| 回到 split-N 加大每核列数 | 错。32→768 列：5041→13688 µs | 1、3 |
| step 被 L0A 32 KB 卡住 | 错。baseK 压到 64（A tile 16 KB）仍越界 | 4 |
| step 被 L1_A 卡住 | 不完整。模型成立（128×1024=131072 正好满），但按模型调参后仍崩 | 4 |
| L0 偏移按固定 `L0A_PP_SIZE` 硬算 | 错。`mmLocalTensors::Init` 只 `SetAddr`，偏移在使用点按 ping-pong 算 | 4 修正 |
| `ceil(step/64) ≤ 2` 是 split-M 的限制 | 错。那是 split-N；split-M 每 AIV 核处理 `step/2`=64 token | 4 修正 |
| 逐 GEMM 放大 step | 错。只放大 1 个 GEMM 也崩；**bf16 路径同样崩** ⇒ step≤128 是全路径硬边界 | 5 |

**结论**：`step ≤ 128` 是这条 kernel 路径的结构性上限，跨量化模式、跨 GEMM 均成立。
1 ms 需要把 passes 从 8 降到 2，唯一路径是重写 `Process()` 的步进循环，
把权重加载提到 step 循环之外。

## 6. 1 ms 的改动清单（下一步实施对象）

见同目录 `PLAN-step-loop-rewrite.md`。
