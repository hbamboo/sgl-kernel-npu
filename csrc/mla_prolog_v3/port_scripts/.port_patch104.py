#!/usr/bin/env python3
"""Record the concurrent down-projection experiment (measured, no benefit, reverted)."""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"
P = ROOT + "/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """| 7 | 两个下投影阶段可重叠/扩核 | mm2 核数 9→12→18（baseN 64→48→32） | 89.75 / 89.58 / **89.35**（T=256）；649.8 / 647.9 / 645.7（T=2048） | **<1%**：阶段耗时不由每核列数决定 |"""
NEW = """| 7 | 两个下投影阶段可重叠/扩核 | mm2 核数 9→12→18（baseN 64→48→32） | 89.75 / 89.58 / **89.35**（T=256）；649.8 / 647.9 / 645.7（T=2048） | **<1%**：阶段耗时不由每核列数决定 |
| 7b | 两阶段**真正并发**（mm1 占 [0,b1)、mm2 占 [b1,b1+b2) 独立 block 区间，一次 kernel 内同时跑） | 新增 `MLA_PROLOG_V3_PARALLEL_DOWN=1`：tiling 把 32 个核分成 16+12（各 48 列、16 对齐），kernel 里 mm2 的偏移与 `UsedBlockParams` 改为相对 `mm2BlockStart` | 同一窗口内：flag 关 73.92 / 74.45 µs，flag 开 74.45 / 74.45（T=256），T=2048 524.4 → 529.2 | **无收益（略差 0.7%）**，与 #7 一致：这两个阶段不是延迟受限。实现完整可用（3 处 tiling + 2 处 kernel），已回滚，脚本留在 `port_scripts/port_patch99..103.py` |"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """## 2. 为什么 L1 是硬的（源码级）"""
NEW2 = """## 1b. 这台机器的测量方差（重要）

同一份二进制、同一脚本，跨窗口实测的差异可达 **±20%**：本会话里 T=256 出现过
`73.9 µs`（某次窗口）与 `89.7 µs`（多次稳定复现），T=2048 对应 `524 µs` 与 `650 µs`。
差异来自共用卡上其它租户的负载（`npu-smi` 可见各卡 4.7 GB 空闲态 ↔ 90 GB 满载态）。
**结论**：所有横向对比必须在同一窗口内交替测量；本文档中的优化结论都是在同窗口 A/B 下得出的，
但绝对数字会随窗口浮动。

## 2. 为什么 L1 是硬的（源码级）"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
io.open(P, "w", encoding="utf-8").write(s)
print("counter-evidence doc updated")
