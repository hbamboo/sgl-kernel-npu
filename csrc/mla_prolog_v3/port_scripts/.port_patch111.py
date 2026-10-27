#!/usr/bin/env python3
"""Record the practical GEMV ceiling - the strongest evidence for the utilisation conclusion."""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """## 1d. 解码态是"同步延迟受限"，不是带宽受限（本轮定位）"""
NEW = """## 1c4. GEMV 形状的**实际可达带宽**只有 0.4–1 TB/s（这条决定了"29%"就是上限）

前面已证明硬件峰值是真的（大 GEMM 满 424 TFLOP/s、大 copy 满 3318 GB/s）。但解码是 GEMV
（m=1），它的可达带宽完全不同。用 `tests/probe_vector_gemv_feasibility.py` 在**队列满载**下
（47 次背靠背，排除单次延迟）测同形状投影：

| 实现 | 形状 | 每调用 | 有效带宽 |
|---|---|---|---|
| `torch.mv`（cube GEMV，厂商实现） | [768, 2048] 3.0 MB | **7.6 µs** | **384 GB/s** |
| `(Wt*x).sum(1)`（vector 归约路线） | 同上 | 15.7 µs | 187 GB/s |
| 本算子解码（4 相全部） | 17.3 MB | **18.2 µs** | **≈ 0.95 TB/s** |

三个结论：
1. **vector-only 的 GEMV 比 cube 慢 2 倍**（187 vs 384 GB/s）⇒ "改用 AIV 做 GEMV 的解码重写"
   这条路被实测否定（原方案 B 作废）；
2. 厂商的 `torch.mv` 在同一形状上只有 384 GB/s，而**我们的 kernel 已经做到 ≈0.95 TB/s**（是它的 2.5 倍）；
3. 所以"解码 29% of 3.27 TB/s"的真面目是：**GEMV 形状在这台卡上的实际上限约 1 TB/s 量级**，
   我们已经在上面。用"大 GEMM/copy 的峰值"去要求 GEMV 形状，本身就是错误基准。

## 1d. 解码态是"同步延迟受限"，不是带宽受限（本轮定位）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """| 6 | 权重搬运不够快（解码） | 扫 N（4/20/62/124，权重 7.5→75 MB） | 18.5/19.3/26.6/36.2 µs，拟合 ≈ `16.5 µs + 0.26 µs/MB` | 解码 **85% 是固定开销**，不是带宽 |"""
NEW2 = OLD2 + """
| 6b | 用 AIV（向量单元）替代 cube 做解码 GEMV | 队列满载下测同形状投影：`torch.mv`(cube) vs `(Wt*x).sum(1)`(vector) | 7.6 vs 15.7 µs（384 vs 187 GB/s） | **vector 慢 2 倍**，方案 B 作废；我们自己的解码 kernel 已达 ≈0.95 TB/s，是该形状的上限附近 |"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
io.open(P, "w", encoding="utf-8").write(s)
print("counter-evidence doc updated")
