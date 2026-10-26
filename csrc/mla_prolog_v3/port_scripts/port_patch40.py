#!/usr/bin/env python3
"""README/report: record the D=192 step cap (64 -> 128) and the He-scaling attribution."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- README-port.md
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 40.9 us | 17.5 us | 5.3 us (BW) | 30% |
| 8 | 41.8 us | 18.4 us | 5.4 us (BW) | 29% |
| 16 | 42.2 us | 18.8 us | 5.6 us (BW) | 30% |
| 64 | 53.9 us | 30.5 us | 7.4 us (BW) | 24% |
| 256 | 93.8 us | 70.4 us | 10.4 us (compute) | 15% |
| 512 | 176.4 us | 153.0 us | 20.8 us (compute) | 14% |
| 2048 | 675.6 us | 652.2 us | 83.6 us (compute) | 13% |"""
NEW = """| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 41.6 us | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.6 us | 18.2 us | 5.4 us (BW) | 30% |
| 16 | 42.7 us | 19.3 us | 5.6 us (BW) | 29% |
| 64 | 53.9 us | 30.5 us | 7.4 us (BW) | 24% |
| 256 | 90.0 us | 66.6 us | 10.4 us (compute) | 16% |
| 512 | 169.4 us | 146.0 us | 20.8 us (compute) | 14% |
| 1024 | 329.3 us | 305.9 us | 41.8 us (compute) | 14% |
| 2048 | 650.9 us | 627.5 us | 83.6 us (compute) | 13% |

Note: this card occasionally flips into a second stable state (~48.5 us instead of
~41.5 us for decode, same binary/config), most likely other tenants on the shared
HBM/L2; all comparisons in this document use same-session A/B runs.

He scaling at T=256 (He=1024/2048/4096/8192 -> 71.9/93.8/143.3/244.5 us) shows the two
down-projection GEMMs (mm1+mm2) account for **~49 us of the ~70 us prefill kernel**
(70%), which is why they are the only meaningful prefill target."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """* `MLA_PROLOG_V3_MAX_STEP=128` (per-step tokens 64 -> 128) is numerically identical and
  ~5% faster at T >= 256, but the 64 cap exists for UB budget reasons on D != 128, so
  it stays an opt-in knob rather than the default."""
NEW2 = """* per-step tokens for D=192 are now **128** by default (upstream: 64).  Validated
  against the CPU golden at T=256/512/1024 and ~4% faster (T=256 93.8 -> 90.0 us,
  T=2048 675.6 -> 650.9 us); 192 tokens/step exceeds the UB budget and faults, so
  every other D keeps 64."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

# ---- REPORT
P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD3 = """| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| 总耗时 (µs) | 41.6 | 41.6 | 53.9 | 93.8 | 176.4 | 342.4 | 675.8 |"""
NEW3 = """| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| 总耗时 (µs) | 41.6 | 41.6 | 53.9 | 90.0 | 169.4 | 329.3 | 650.9 |"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
OLD4 = """| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| kernel (µs) | 18.2 | 18.2 | 30.5 | 70.4 | 153 | 319 | 652 |"""
if OLD4 in s:
    s = s.replace(OLD4, """| kernel (µs) | 18.2 | 18.2 | 30.5 | 66.6 | 146 | 306 | 627 |""")
OLD5 = """已排除的优化项（都有实测数据）：核数扫描（32/24/16/8/4）、每步 token 上限、"""
NEW5 = """已排除的优化项（都有实测数据）：核数扫描（32/24/16/8/4）、"""
assert s.count(OLD5) == 1
s = s.replace(OLD5, NEW5)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
