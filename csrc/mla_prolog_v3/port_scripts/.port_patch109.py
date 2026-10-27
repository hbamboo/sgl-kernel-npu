#!/usr/bin/env python3
"""Record the hardware calibration and the step>128 overflow boundary."""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """## 1d. 解码态是"同步延迟受限"，不是带宽受限（本轮定位）"""
NEW = """## 1c2. 硬件峰值是**真的可达**（先排掉"峰值虚标"这个可能）

`tests/probe_practical_peak.py`（同一张卡、同一容器）：

| 项目 | 实测 | 占标称 |
|---|---|---|
| bf16 GEMM 2048³ | 383 TFLOP/s | 90% |
| bf16 GEMM 4096³ | **423.5 TFLOP/s** | **100%** |
| bf16 GEMM 8192³ | 430.8 TFLOP/s | 102% |
| HBM copy（4 GiB） | **3318 GB/s** | **101%** |

⇒ 标称的 424 TFLOP/s 与 3.27 TB/s **不是虚标**，大 GEMM 能跑满。因此本算子的 12–16% 利用率
不是"硬件达不到"，而是**算子形状（m=1 解码 / m≤128 预填）与 kernel 流水结构**造成的结果——
这既让结论更硬，也说明"提高利用率"必须改变 m 的组织方式（见 §1d 与 `DESIGN-kernel-rewrite.md`）。

## 1c3. 每步 token 数卡在 128（step>128 触发 507015）

`MLA_PROLOG_V3_MAX_STEP` 实测：128 → 89.73 µs；**192 / 256 → 设备错误 507015（device error type 3）**。
按 L1 容量手算：m=192 时 `stepK = L1_A/(m·baseK·2) = 2`、`kL1Step=256`、A=192×256×2=96 KB（未超 128 KB），
所以溢出点不在 A 侧；要放开必须逐项排查 mm1/mm3 的 B 侧与 workspace 分配——这属于"再改一处就要连带
排查一片"的典型，收益是预填每 step 的固定开销（按 T 扫描约 15 µs / 17%）。

## 1d. 解码态是"同步延迟受限"，不是带宽受限（本轮定位）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("counter-evidence doc updated")
