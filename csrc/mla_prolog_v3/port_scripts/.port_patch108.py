#!/usr/bin/env python3
"""Record the decode diagnosis: sync-latency bound, not bandwidth bound."""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """## 2. 为什么 L1 是硬的（源码级）"""
NEW = """## 1d. 解码态是"同步延迟受限"，不是带宽受限（本轮定位）

两个探针把结论钉住了：

**探针 1（权重无关）** `tests/probe_decode_scaling.py`：T=1 下同时扫 He(1024/2048) × N(2/20)，
权重从 3.75 MB 到 16.5 MB：

| He | N | 权重 (MB) | 单次调用 (µs) |
|---|---|---|---|
| 1024 | 2 | 3.75 | 41.25 |
| 1024 | 20 | 13.88 | 39.70 |
| 2048 | 2 | 6.38 | 38.62 |
| 2048 | 20 | 16.50 | 41.51 |

拟合斜率 **0.02 µs/MB ≈ 0**：把权重放大 4.4 倍，耗时不变。⇒ 解码耗时**与要搬的权重无关**，
16.5 MB 全部"免费"地藏在固定开销里。

**探针 2（框架地板）** `tests/probe_kernel_overhead.py`：同栈上的平凡 kernel（helloworld，8 元素）
单次调用 **22.5 µs**，我们的解码 41.8 µs ⇒ 差 19 µs 才是"我们自己的部分"；而稳态（流水满载）实测
18.6 µs/call，说明主机能把 18.6 µs 排空，**kernel 就是瓶颈且正好等于它的纯开销**。

**结论**：解码 kernel 的 18.2 µs 里几乎没有"搬数据"或"算"的成分，而是 **4 个阶段 × 每阶段若干次
跨核 barrier（`CrossCoreSetFlag/WaitFlag`）的往返延迟**。这也解释了为什么：核数（8→32）、权重
规模（3.75→16.5 MB）、量化模式全都改不动它。
**这意味着**：要压到 ~10 µs，必须减少阶段数/barrier 往返（而不是优化搬运或换算法）；
按 4 阶段、每阶段约 4 µs 计，把 4 段合成 2 段大致能到 10–12 µs ⇒ 稳态 18.6 → ~12 µs/call，
每 step 省约 0.3 ms。

## 2. 为什么 L1 是硬的（源码级）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)

P2 = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/DESIGN-kernel-rewrite.md"
s = io.open(P2, encoding="utf-8").read()
OLD2 = """## 方案 B：解码态持久 kernel（收益最大，解码 18.2 → 8~10 µs）"""
NEW2 = """## 方案 B′（首要）：解码态"少阶段"kernel —— 针对同步延迟，而非带宽

**先决事实**（`PERF-COUNTEREVIDENCE.md` §1d）：解码耗时与权重字节数无关（3.75→16.5 MB 时延不变，
斜率 0.02 µs/MB），也与核数、量化模式无关；18.2 µs 基本是 **4 个阶段各自的跨核 barrier 往返**。

因此改写要点不是"用 AIV 做 GEMV"（那是带宽视角），而是**减少阶段数与 barrier 次数**：

| 现在 | 目标 | 手段 |
|---|---|---|
| mm1 → 向量(rmsnorm/quant) → mm3(mmQcQr) → mm4(mmQn) → 向量(rope/scatter) = 4–5 段 | 2 段 | ① 把 ckv/kr 的 rmsnorm 与 cq 的 rmsnorm 合并到同一次向量段；② 把 mm3 与 mm4 之间用"单核内串行 + 一次 barrier"替代逐段 barrier；③ 全 T≤16 时只用 16 个 block，减少 barrier 参与方 |

验收门与回退条件同下（golden 五路 ≥0.9999、T≥64 走旧路径且必须逐位一致、失败即回退）。
预期：4 段 → 2 段 ⇒ 18.2 → 10–12 µs，稳态 18.6 → ~12 µs/call，每 step 约省 0.3 ms。

## 方案 B：解码态持久 kernel（若带宽成为瓶颈再做）"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
io.open(P2, "w", encoding="utf-8").write(s)
print("docs: decode diagnosis + revised plan")
