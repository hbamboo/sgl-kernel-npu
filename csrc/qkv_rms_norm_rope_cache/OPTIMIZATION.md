# qkv_rms_norm_rope_cache 优化记录

本文记录从「迁移初版」到当前实现的所有优化项：原理、实现要点、实测收益，以及被否决的方案。
配套数据与脚本见 `benchmark/qkv_rms_norm_rope_cache/`。

## 1. 测量口径

| 项 | 说明 |
| --- | --- |
| 设备 | Ascend950DT 单卡（32 AIC / 64 AIV），CANN 9.1.0，torch 2.10 + torch_npu 2.10 |
| 主形状 | tokens = 34460（另测 32768 / 8192），head_dim = 128，q/k/v = 32/8/8 头 |
| 缓存 | PA_NZ：`k_cache/v_cache = [270, 64, 128, 16]` bf16，`index = arange(tokens)` |
| 计时 | `npu.Event` 包整次 Python 调用（warmup 20 / iters 50 取中位数）；profiler 实测 host 开销 AscendC ≈40 us、Triton ≈72 us，相对 0.6/5.0 ms 可忽略 |
| 有效访存 | 864.5 MB（读 qkv+cos/sin 440.7 MB，写 q_out+k/v cache 423.8 MB），两种实现相同 |
| Roofline | 同设备 805 MB 纯 copy = 251 us → **3208 GB/s** |

```bash
python benchmark/qkv_rms_norm_rope_cache/bench_vs_triton.py --tokens 34460 --warmup 20 --iters 50
```

## 2. 总览

| 阶段 | 改动 | 32k tokens | 34460 tokens | 有效带宽 |
| --- | --- | ---: | ---: | ---: |
| 迁移初版 | UB-tile kernel，逐 head 256B DMA | 6 282 us | 5 152 us | 168 GB/s |
| A2 | 去逐 head 同步 + 按 token 批量取数 | 5 896 us | — | — |
| A3 | UB tile 减半 | 5 556 us | — | — |
| A4 | 去重复 Cast + Rope 死拷贝 | 4 908 us | — | — |
| B1 | **RegBase 重写** | — | 2 250 us | 384 GB/s |
| B2 | 单 VEC_SCOPE + gamma/cos-sin 复用 | — | 1 047 us | 825 GB/s |
| B3 | 连续 slot 批量 scatter + ubRows=192 | — | 737 us | 1 173 GB/s |
| B4 | gamma 折进 RoPE 系数 | — | 699 us | 1 236 GB/s |
| B5 | 核数按实测工作量重分（39/16/9） | 566.7 us | **597.0 us** | **1 448 GB/s** |
| 对照 | Triton `split_qkv_rmsnorm_rope` | 4 788 us | 5 031 us | 172 GB/s |

**最终：比迁移初版快 8.6x，比 Triton 快 8.4x。**

---

## 3. 阶段 A：UB-tile 版本的优化（已被 RegBase 取代，过程仍有效）

### A1. 修正 tiling 的 UB 公式（正确性）

- **原理**：kernel 的 `InitBuffer()` 为 Q/K 分支分配 `inQueueX(×2) + inQueueY(×2) + locBuf0(×2) + locBuf1 + locBuf2(×2) + locBuf3 + outQueue(×2)`，但移植版 tiling 只按 `inX + inY + 4×locBuf + outQueue` 估算每行占用，漏掉了 locBuf0/locBuf2 的 double buffer，`ubFactor` 偏大 ~1.29 倍。
- **收益**：修复前 kernel 一启动就 UB 越界，报 `device error 507035`；修复后可用（这是正确性修复，不是性能项）。

### A2. 去掉逐 head 同步 + 按 token 批量取数

- **原理**：`SplitQkv` 原来每个 head 发一次 256 B `DataCopyPad`，且循环内带 `SetWaitFlag<S_MTE2>`；scatter 每行还有 `SToMTE3Sync()`。1.57 M 个 head-vector 被串行化成小 DMA。改动：同一 token 的连续 head 在 GM/UB 都连续 → 一次 `DataCopyPad` 取一整段；`index`/`page`/`offset` 每 token 只算一次（原来每 head 3 次 int64 除法）；去掉逐行同步。
- **收益**：32k 6 282 → 5 896 us（**-6.2%**）。消融显示单去掉逐 head 除法无收益，瓶颈是小 DMA 的串行化。

### A3. UB tile 减半

- **原理**：A950 上把 `ubFactor` 从填满 UB（54 行）降到一半（27 行）反而更快，工作集小、跨迭代依赖短。
- **收益**：5 896 → 5 556 us（**-5.8%**）。

### A4. 去重复 Cast + Rope 末尾死拷贝

- **原理**：Q/K 路径把同一份 bf16 输入 Cast 了两次（`xLocalFp32` 和 `dstLocalFp32`），后者可直接由 `RmsNorm` 从源行读取；`Rope` 结束时把结果又抄回死的 cos/sin buffer。
- **收益**：5 555 → 4 905 us（**-11.7%**，两个整 tile 向量操作/步）。

---

## 4. 阶段 B：RegBase 重写（当前实现）

参考 `ops-transformer/posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/` 的做法：A5 上用 RegBase（MicroAPI）把计算放在向量寄存器里，而不是每个算子都读写一遍 UB。

### B1. RegBase 重写：每 head 只 1 次 UB 读 + 1 次 UB 写

- **原理**：UB-tile 版本每个 head 要过约 10 遍 UB（cast→平方→归约→除法→gamma→rope 的 4 个乘、1 个加…），UB 往返成为瓶颈。RegBase 版把 head_dim=128（= 2×VL_FP32，正好两个整寄存器、无 mask/tail）的整段计算放进 `__VEC_SCOPE__`，用 `RegTensor<float>` 在寄存器间传递，只保留一次读入和一次写出。
- **实现**：`qkv_rms_norm_rope_cache_regbase.h`（`RbNormRopeTile` + `KernelQkvRmsNormRopeCacheRegbase`），kernel 入口 `qkv_rms_norm_rope_cache.cpp` 改为调用它。
- **收益**：5 152 → 2 250 us（**2.3x**），带宽 168 → 384 GB/s。

### B2. 单 VEC_SCOPE + gamma/cos-sin 复用

- **原理**：B1 是「每个 head 开一个 `__VEC_SCOPE__`」，每个 scope 都是流水线边界。改成**整个 UB tile 一个 VEC_SCOPE**，并把 `gamma` 提到循环外（每 tile 只载入一次）、`cos/sin` 每 token 载入一次并被该 token 的所有 head 复用。
- **收益**：2 250 → 1 047 us（**2.15x**）。

### B3. 连续 slot 的批量 scatter + ubRows=192

- **原理**：PA_NZ 下单个 head 的 128 维被切成 8 段 32 B，段间间隔 `blockSize*32B`，所以逐 head 写是 8 个 32 B 小突发。但**同一 page 内连续 token 的同一 `(head, D/16)` 块在 cache 上是连续的**：`index` 连续时，n 个 token 的这块可以一次 `n*32B` 连续写出。
- **实现**：`Scatter` 先找「同一 page 内 slot 连续」的运行段；n>1 时走 `ScatterRun`：`DataCopy`（**32 B block 单位**）一次写 n 个 token（`blockCount=n, blockLen=1, srcStride=numHeads*8-1, dstStride=0`）。注意 `DataCopyPad` 的字节 stride 语义在这里会越界（实测 `device error 507035`）。
- **收益**：scatter 开销 ~465 → ~210 us，总时间 1 047 → 737 us（**-30%**）。同时 ubFactor 128 → 192（受 UB 预算限制：输入/输出各 2 个 buffer + cos/sin staging，且每 tile 最多 64 个 token 否则小 head 数形状会撑爆 UB）。

### B4. gamma 折进 RoPE 系数

- **原理**：`y = x/rms*gamma`，RoPE 是 `y*cos ± rot(y)*sin`。把 gamma 预先乘进每 token 的 cos/sin（4 次乘/token，摊到该 token 的所有 head），每 head 省掉 2 次整向量乘。
- **实现要点**：①原地折叠（`Mul(c0,c0,g0)` …）保持 13 个 `RegTensor`，新增 4 个寄存器会超出向量寄存器数量、结果直接算错；②**rotate-half 的交叉项要配另一半的 gamma**：`y_hi*sin_lo` 里的 sin 系数应乘 `g_hi`，即 `s0 *= g_hi`、`s1 *= g_lo`（一开始配反，q_out 全错）。
- **收益**：737 → 699 us（**-5%**）。

### B5. 核数按实测工作量重分（Q:K:V = 39:16:9）

- **原理**：Q/K 每 head 跑完整 norm+RoPE（约 20 条向量指令/head），V 在 RegBase 版里是纯 identity scatter（无向量计算），但 Q/K/V 三组都要付 PA_NZ scatter。上游按 head 数分核（K:V = 0.6:0.4），在 A5 上把 K 饿着了——K 每 head 成本约为 Q 的 1.7 倍。
- **实现**：先按工作量给 KV 预算，再在 KV 内按权重分：`kWorkQ=1.0, kWorkK=1.7, kWorkV=0.9`（由 64 核 sweep 定标）。本形状得到 Q=39 / K=16 / V=9。
- **收益**：699 → 597 us（**-15%**）。sweep 数据：43/13/8（上游）=703 us，39/16/9=**596.5 us**，47/11/6=802 us（K 不够）。

### B6. V 路径直接 scatter

- **原理**：V 是纯拷贝，原实现先 `load → 寄存器 → store` 到输出 UB 再 scatter；改为直接从输入 staging buffer scatter，用 `MTE2_MTE3` / `MTE3_MTE2` 显式同步。
- **收益**：整机时间无可测收益（Q 是关键路径），但省掉 V 组每 head 4 条向量指令、简化了路径，并为 B5 的分核留出空间。

---

## 5. 被否决的方案（均已实测）

| 方案 | 结果 | 原因 |
| --- | --- | --- |
| 合并 K/V 成一组（每核先 K 后 V，mode 1） | 607~614 us（+2%） | 顺序执行，K 的向量与 V 的 DMA 没真正重叠 |
| V 挂到 Q 核上（mode 2） | 656 us（+9%） | V 的 scatter 串到 Q 关键路径尾部，而三组时 V 是并行的 |
| 按 head 数分核（上游比例） | 703 us（+18%） | K 每 head 更贵，被饿着 |
| 只按向量量分核（V 权重 0.25） | 802 us（+34%） | 同上，K 分得太少 |
| 输出队列加深到 4 | 无收益且破坏正确性 | TQue 只跟踪 V，MTE3 未完成时 buffer 被复用 |
| ubRows > 192 | 无法实现 | UB 预算（in/out 各 2 buffer + cos/sin staging）不够 |

---

## 6. 极限分析

- **向量吞吐**：Q 组 39 核，每核约 1.1 M 条向量指令，597 us / 1.1 M ≈ **1 条 VL 指令/cycle**（fp32 64 lane）——向量单元已满吞吐。
- **理论地板**：总向量工作（Q+K 共 55 M 条 VL）在 64 核上 ≈ 478 us；K/V 的 scatter DMA（134 MB）已用批量写与计算重叠。当前 597 us 已落在「向量地板 + 未完全重叠的 scatter」区间。
- **与 memcpy roofline 的差距**（1448 vs 3208 GB/s）来自算法固有的 fp32 向量指令数（≈20 条/head：2 次 cast、平方、归约、4 条标量、2 次除法、6 条 RoPE、2 次 store），不降精度无法再省。
- **已无进一步明显收益的方向**：三种分组策略都已试过（三组最优），tile 尺寸受 UB 限制，scatter 已批量。

## 7. 精度与验证

- `tests/python/sgl_kernel_npu/test_qkv_rms_norm_rope_cache.py`：**10 passed**（8 组 shape + 全 `index=-1` + 重复执行一致性；golden 为上游 ST 公式的 fp32 复算）。
- 与 Triton 同输入对跑（34460 / 32768 / 8192）：`q_out`、`k_cache` **0 offender**（>1 bf16 ULP 且 |diff|>1e-5），`v_cache` **逐 bit 相等**。
- 复用 `qkv_rms_norm_rope_cache_b16_pa_nz_quant.h`/`_comm.h` 的旧 UB-tile 实现已删除（未被编译），备份在 `/mnt/paas/hxdnpu/removed_kernel_backup/`，上游源码仍在 `ops-transformer/posembedding/qkv_rms_norm_rope_cache/`。

## 8. 复现

```bash
# 构建（A5）
cd sgl-kernel-npu && ./build.sh -a kernels Ascend950PR_9599
pip install --no-deps output/sgl_kernel_npu-*.whl

# 精度自测
pytest -q tests/python/sgl_kernel_npu/test_qkv_rms_norm_rope_cache.py

# 精度互检 + 性能对比
python benchmark/qkv_rms_norm_rope_cache/bench_vs_triton.py --tokens 34460 --warmup 20 --iters 50
```
