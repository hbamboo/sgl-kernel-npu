#!/usr/bin/env python3
"""Record the step>128 root cause and why unlocking it would not help."""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """## 1c3. 每步 token 数卡在 128（step>128 触发 507015）

`MLA_PROLOG_V3_MAX_STEP` 实测：128 → 89.73 µs；**192 / 256 → 设备错误 507015（device error type 3）**。
按 L1 容量手算：m=192 时 `stepK = L1_A/(m·baseK·2) = 2`、`kL1Step=256`、A=192×256×2=96 KB（未超 128 KB），
所以溢出点不在 A 侧；要放开必须逐项排查 mm1/mm3 的 B 侧与 workspace 分配——这属于"再改一处就要连带
排查一片"的典型，收益是预填每 step 的固定开销（按 T 扫描约 15 µs / 17%）。"""
NEW = """## 1c3. 每步 token 数卡在 128：根因已定位，且放开它**没有收益**

**边界实测**（`tests/probe_step_overflow.py`，每例独立进程，避免设备错误污染上下文）：

| T | N | step | 结果 |
|---|---|---|---|
| 192 | 20 | 192 | FAIL 507015 |
| 192 | 4 | 192 | FAIL 507015（**与 N 无关** ⇒ 不是 mm3/mm4）|
| 192 | 4 | 192（关 query_norm） | FAIL 507015 |
| 144 | 20 | 144 | FAIL 507015 |
| 64 | 20 | 64 | OK |
| 128 | 20 | 128 | OK |
| 256 | 4 | 256 | FAIL 507015 |

失败/成功的分界正好是 **`ceil(step / 64 个 AIV 核)`：≤2 可以，=3 即崩**。向量侧每个 AIV 核要缓存
`ceil(step/vectorBlockNum)` 个 token 的 UB，UB 预算只放得下 2 个 —— 这与 N、mm1 的 A 侧 L1（m=192 时
仅 96 KB，未超 128 KB）都无关，因此**根因在向量段的 UB 分配**，不是 cube。

**但放开它没有收益**：把 T 扫描按"每步 token 数"归一后可以看到，
`t ≈ 9.4 µs（kernel 固定）+ 0.31 µs/token`，即每步的成本**基本与 token 数成正比**；
step=256 只是把两步合成一步，步内成本翻倍，总时长不变（T=256 现在 89.7 µs，两步骤 10+2×39.9）。
真正能省的只有那 ~10 µs 的 kernel 固定开销，而它已经在 §1d 里被归入"每次调用的固定成本"。

结论：这条线**不必再挖**——它既需要重构向量段的 UB 分块，又换不来可测量的时间。"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("counter-evidence doc updated")
