# qkv_rms_norm_rope_cache 性能对比（AscendC 迁移版 vs Triton split_qkv_rmsnorm_rope_kernel）

> 说明：本文前半部分是历史记录（UB-tile 版本的测量）。最终实现是 RegBase 版，
> 结论看文末「优化收敛记录」；优化项的原理与收益见 `csrc/qkv_rms_norm_rope_cache/OPTIMIZATION.md`。

日期：2026-10-21　设备：Ascend950DT 单卡（ASCEND_RT_VISIBLE_DEVICES=7，32 AIC / 64 AIV）
CANN 9.1.0，torch 2.10.0 + torch_npu 2.10.0.post6，triton-ascend 3.2.0

## 目标形状（按用户给定）

| 输入 | shape |
| --- | --- |
| qkv | `[34460, 6144]`（q 32 头 + k 8 头 + v 8 头，head_dim 128） |
| sin | `[34460, 1, 1, 128]` |
| cos | `[34460, 1, 1, 128]` |
| index | `[34460]`，= `arange(34460)`（连续槽位） |
| k_cache / v_cache | `[270, 8*8, 128, 16]` bf16（PA_NZ） |

有效访存 ≈ 864.5 MB（读 440.7 MB + 写 423.8 MB），两种实现相同。

## 对比结果

| 实现 | median | mean | min | 有效带宽 |
| --- | ---: | ---: | ---: | ---: |
| **AscendC `torch.ops.npu.qkv_rms_norm_rope_cache`** | **5151.7 us** | 5153.0 us | 5147.1 us | 167.8 GB/s |
| **Triton `split_qkv_rmsnorm_rope`** | **5027.3 us** | 5027.9 us | 5016.0 us | 172.0 GB/s |

两者相差 **1.02×（AscendC 慢 2.5%）**，可以认为打平。

> 口径说明：AscendC 版按 PA_NZ 布局 scatter 写 k_cache/v_cache（离散写）；
> Triton 版输出 q/k/v 三个 ND 张量（连续写）。写出的字节数相同，但 AscendC 的写更碎。

其他尺寸（同一构建、同一脚本）：

| tokens | AscendC | Triton | 比值 |
| ---: | ---: | ---: | ---: |
| 1 024 | 192.0 us | 170.8 us | 1.12× |
| 8 192 | 1 256.9 us | 1 248.6 us | 1.01× |
| 32 768 | 4 910.9 us | 4 787.9 us | 1.03× |
| 34 460 | 5 151.7 us | 5 027.3 us | 1.02× |

参考 roofline：同设备 805 MB 纯 copy = 251 us = **3 208 GB/s**。两者都只跑到 ~5%，
说明都还是向量指令吞吐受限，不是 HBM 受限。

## 测量口径：kernel-only vs 端到端（torch_npu profiler trace）

用 `bench_kernel_time.py` 导出 chrome trace，按设备事件聚合（每 call，iters=10）：

| | AscendC | Triton |
| --- | ---: | ---: |
| host enqueue（不含设备执行） | 39.6 us/call | 72.1 us/call |
| 设备 kernel 事件 | **5119.7 us**（`qkv_rms_norm_rope_cache_1`） | **4951.0 us**（`split_qkv_rmsnorm_rope_kernel`） |
| 端到端（host+device） | 5128.2 us/call | 4960.4 us/call |

两者 host 开销都只有几十 us，**5 ms 基本都是设备 kernel 时间**；AscendC 比 Triton 慢约
3.4%（5119.7 vs 4951.0）。此前用 `npu.Event` 包整次调用的结果（5151.7 / 5027.3 us）
与之一致，差异在测量噪声内。

## 精度验证（先于性能对比执行）

1. `tests/python/sgl_kernel_npu/test_qkv_rms_norm_rope_cache.py`：**10 passed**
   （8 组 shape + 全 index=-1 + 重复执行一致性；golden 为上游 ST 公式的 float32 复算）。
2. 34460 形状下与 Triton 对跑（同一份输入）：

   | 输出 | max_abs | 超过 1 bf16 ULP 且 \|diff\|>1e-5 的元素 |
   | --- | ---: | ---: |
   | q_out | 3.125e-2（= \|x\|≈4 处 1 ULP） | 0 / 141,148,160 |
   | k_cache | 3.125e-2 | 0 / 35,389,440 |
   | v_cache | 0（逐 bit 相等） | 0 / 35,389,440 |

## 本轮优化（6282 us → 4908 us @32k，-21.9%）

按消融定位后依次实施（每步都跑精度）：

| # | 改动 | 说明 |
| --- | --- | --- |
| 1 | `SplitQkv` 按 token 批量取数 | 同一 token 的连续 head 在 GM/UB 都连续，一次 `DataCopyPad` 取一整段，取代逐 head 256 B 拷贝 + 每 head 一次 `S→MTE2` 同步 |
| 2 | scatter 去掉逐 head 除法/取模 | `index` 每 token 读一次、`page/offset` 每 token 算一次；去掉逐行 `S→MTE3` 同步 |
| 3 | cos/sin 用增量方式算 token 下标 | 去掉逐行 int64 除法 |
| 4 | 去掉重复的 bf16→fp32 Cast | `RmsNorm` 直接对源行做除法，调用方不再预拷一份 x 到 dst |
| 5 | 去掉 Rope 末尾的无用回拷贝 | 结果已在目标 buffer，旧代码又抄回已死的 cos/sin buffer |
| 6 | UB tile 减半 | A950 上小 tile 更快：满 tile 5264 us → 半 tile 4908 us（32k）。该参数随 UB-tile kernel 一起删除，RegBase 版直接固定 ubRows=192 |

消融数据（32k，自上而下逐步剥离）：

| 状态 | 时间 |
| --- | ---: |
| 迁移初版 | 6 282 us |
| 去掉逐 head 同步 + 批量取数 | 5 896 us |
| 再去掉 97% cos/sin 拷贝（实验） | 5 443 us |
| 再跳过 RmsNorm 两个逐行循环（实验） | 3 482 us |
| 最终（#4、#5、#6 等） | 4 908 us |

剩余瓶颈：Q/K 路径仍约有 11 条 fp32 向量指令/element（RmsNorm 的逐行 `Div` 与
逐行 gamma `Mul` 各占 1 条/行，rope 约 4.5 条/element）。要把 4.9 ms 继续压下去，
需要做行导向的重写（一次处理整行 6144 元素、把逐行广播指令改成整 tile 指令）。

## 复现

```bash
cd sgl-kernel-npu/benchmark/qkv_rms_norm_rope_cache
python bench_vs_triton.py --tokens 34460 --warmup 20 --iters 50
python bench_vs_triton.py --tokens 32768 --warmup 20 --iters 50
python bench_vs_triton.py --tokens 34460 --iters 20 --index-mode skip   # 去掉 scatter 的对照
```

需要先把本次构建的 wheel 装进环境（`pip install --no-deps output/sgl_kernel_npu-*.whl`）。

## RegBase 重写后的最终对比（34460 token）

| 实现 | median | 有效带宽 | 相对 Triton |
| --- | ---: | ---: | ---: |
| AscendC（RegBase + 批量 scatter + gamma 折叠 + 核数重分 39/16/9） | **598.7 us** | 1444 GB/s | **快 8.40x** |
| Triton `split_qkv_rmsnorm_rope` | 5032.6 us | 172 GB/s | 1.00x |

其他尺寸（同一构建）：

| tokens | AscendC | Triton | 比值 |
| ---: | ---: | ---: | ---: |
| 8 192 | 167.5 us | 1 236.1 us | 7.4x |
| 32 768 | 566.7 us | 4 788.8 us | 8.5x |
| 34 460 | 598.7 us | 5 028.5 us | 8.4x |

演进：迁移初版 5 152 us（168 GB/s）-> RegBase 2 250 us -> 单 VEC_SCOPE/复用 1 047 us
-> 批量 scatter + ubRows=192 737 us -> gamma 折叠 699 us -> 核数按实测工作量重分（Q:K:V=38:16:10，K 权重 1.7）**600.5 us / 1440 GB/s，约为 memcpy roofline 3208 GB/s 的 45%**。
仍有关键路径余量：K 组（norm+RoPE+scatter）在 16 核下已是瓶颈，扫 64 核分配得到的最优点即 39:16:9。

精度：`test_qkv_rms_norm_rope_cache.py` 10 passed；与 Triton 对跑 0 offender，v_cache 逐 bit 相等。

## 优化收敛记录（最终）

| 阶段 | 32k token | 34460 token | 有效带宽(34460) |
| --- | ---: | ---: | ---: |
| 迁移初版（UB tile） | 6 282 us | 5 152 us | 168 GB/s |
| 去掉逐 head 同步/批量取数/UB tile 半化 | 4 908 us | — | — |
| RegBase 重写（v1） | — | 2 250 us | 384 GB/s |
| 单 VEC_SCOPE/整 tile + gamma·cos-sin 复用 | — | 1 047 us | 825 GB/s |
| 连续 slot 批量 scatter + ubRows=192 | — | 737 us | 1 173 GB/s |
| gamma 折进 RoPE 系数 | — | 699 us | 1 236 GB/s |
| 核数按实测工作量重分（Q:K:V = 39:16:9） | 566.7 us | **598.8 us** | **1 444 GB/s** |
| Triton `split_qkv_rmsnorm_rope`（对照） | 4 788 us | 5 024 us | 172 GB/s |

**最终：比迁移初版快 8.6x，比 Triton 快 8.4x。**

已尝试但更差、已否决的方案：
- 合并 K/V 为一组（每核先 K 后 V）：+2%；
- 把 V 挂到 Q 核上（模式 2，顺序执行）：+9%（V 的 scatter 串到 Q 关键路径上）；
- 按 head 数或按纯向量工作量分核（不给 K 加权）：+8~30%。

极限判断：
- Q 组（39 核）每核约 1.1M 条向量指令，实测 ≈**1 条 VL 指令/cycle**（fp32 64 lane），向量单元满吞吐；
- 总向量工作 55M 条 VL 在 64 核的理论地板 ≈ 478 us；K/V 的 scatter DMA（134 MB）已用批量写与计算重叠；
- 当前 598.8 us 已落在“向量地板 + 未完全重叠的 scatter”区间内，剩余与 memcpy roofline（3208 GB/s）的差距来自算法固有的 fp32 向量指令数（≈20 条/head）。

## 精度（最终构建）

`pytest test_qkv_rms_norm_rope_cache.py` 10 passed；34460 / 32768 / 8192 三个尺寸与 Triton 对跑
`q_out`、`k_cache` 均 0 offender（≤1 bf16 ULP），`v_cache` 逐 bit 相等。
