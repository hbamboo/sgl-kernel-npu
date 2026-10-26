#!/usr/bin/env python3
"""Docs: record the out-variant (13-14% decode) and refresh numbers."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- README-port.md
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """* [x] host overhead reduced (tiling cache + workspace reuse)"""
NEW = """* [x] host overhead reduced (tiling cache + workspace reuse)
* [x] `mla_prolog_v3_out` entry point (caller-provided output buffers): **bit-identical**
      to the allocating entry and **13-14% faster at decode** (T=1: 41.8 -> 36.3 us,
      T=8: 42.4 -> 36.3 us) because the five `at::empty` calls disappear; no change at
      T=256.  This is also the entry point the serving stack should use: one set of
      buffers per layer, stable addresses for ACL graph capture."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """| 1 | 41.6 us | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.6 us | 18.2 us | 5.4 us (BW) | 30% |"""
NEW2 = """| 1 | 41.6 us (36.3 with `_out`) | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.6 us (36.3 with `_out`) | 18.2 us | 5.4 us (BW) | 30% |"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

# ---- REPORT
P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()

OLD3 = """| 调用 | `torch.ops.npu.mla_prolog_v3(...)`（Direct kernel launch，不经 aclnn/GE） |"""
NEW3 = """| 调用 | `torch.ops.npu.mla_prolog_v3(...)`（分配输出）/ `torch.ops.npu.mla_prolog_v3_out(...)`（调用方给缓冲区，解码快 13–14%）|"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)

OLD4 = """优化轨迹（T=1）：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希 key）
→ 41.6 µs（核数启发式：T≤128 用 16 核）；预填 T=256：118.8 → **93.8 µs**（mm1 最小瓦片 64→32）。"""
NEW4 = """优化轨迹（T=1）：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希 key）
→ 41.6（核数启发式：T≤128 用 16 核）→ **36.3 µs**（`_out` 变体，调用方预分配缓冲，逐位一致）；
预填 T=256：118.8 → **90.0 µs**（mm1 最小瓦片 64→32 + D=192 每步 128 token）；T=2048：687.7 → **650.9 µs**。"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
