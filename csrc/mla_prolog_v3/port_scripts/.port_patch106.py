#!/usr/bin/env python3
"""Report + counter-evidence: steady-state serving cost is kernel-bound.

Adds the layer-loop measurement (18.6 us/call at T=1 -> 0.87 ms per 47-layer decode step) and
corrects the earlier emphasis on host-side optimisations: they cut single-call *latency* but
not throughput, because with the queue full the kernel dominates.
"""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"

P = ROOT + "/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()
OLD = """## 2. 为什么 L1 是硬的（源码级）"""
NEW = """## 1c. 稳态（流水线满载）时是 kernel 说了算

单次调用的 33 µs 是**延迟**（含主机入队与事件）；服务里一层的 prologue 是背靠背下发的，
所以真正要看的是吞吐。`tests/bench_layer_loop.py` 连发 47 层（一次 decode step 的层数）：

| T | 每次调用 | 47 层（一个 step） |
|---|---|---|
| 1 | **18.6 µs**（torch.ops）/ 18.3（PyBind） | **0.87 ms** / 0.86 |
| 8 | 18.8 / 18.9 | 0.88 / 0.89 |
| 64 | 31.1 / 31.5 | 1.46 / 1.48 |

两个结论：
1. 稳态下 **PyBind / `_out` 与 `torch.ops` 的差别消失**（都被 kernel 掩盖）——它们只降低单次
   延迟，不改变吞吐；真正的成本是 kernel 的 18.6 µs/层；
2. 解码态的 prologue 每个 step 约 **0.87 ms**（47 层），这正是"改写解码 kernel（18.2 → 8~10 µs）"
   能拿到的量（每 step 省 ~0.4 ms）。

## 2. 为什么 L1 是硬的（源码级）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)

P2 = ROOT + "/csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md"
s = io.open(P2, encoding="utf-8").read()
OLD2 = """调用路径（T=1）：分配版 41.6 µs、`_out` 36.2 µs（−13%）、**PyBind 32.9 µs（−21%）**。"""
NEW2 = """调用路径（T=1，单次延迟）：分配版 41.6 µs、`_out` 36.2 µs（−13%）、**PyBind 32.9 µs（−21%）**。

**稳态吞吐（47 层背靠背，一次 decode step）**：每次调用 **18.6 µs**（T=1）/ 31.1 µs（T=64），
一个 step 合计 **0.87 ms / 1.46 ms**；此时三条入口无差别——成本由 kernel 决定，主机优化只降低
单次延迟。"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
io.open(P2, "w", encoding="utf-8").write(s)
print("docs updated with the steady-state picture")
