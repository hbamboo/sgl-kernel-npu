#!/usr/bin/env python3
"""Refresh README timings after the mm1 min-tile default change."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 41.6 us | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.7 us | 18.3 us | 5.4 us (BW) | 29% |
| 16 | 41.4 us | 18.0 us | 5.6 us (BW) | 31% |
| 64 | 54.3 us | 30.9 us | 7.4 us (BW) | 24% |
| 256 | 96.0 us | 72.6 us | 10.4 us (compute) | 14% |
| 2048 | 688.7 us | 665.3 us | 83.6 us (compute) | 13% |"""
NEW = """| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 40.9 us | 17.5 us | 5.3 us (BW) | 30% |
| 8 | 41.8 us | 18.4 us | 5.4 us (BW) | 29% |
| 16 | 42.2 us | 18.8 us | 5.6 us (BW) | 30% |
| 64 | 53.9 us | 30.5 us | 7.4 us (BW) | 24% |
| 256 | 93.8 us | 70.4 us | 10.4 us (compute) | 15% |
| 512 | 176.4 us | 153.0 us | 20.8 us (compute) | 14% |
| 2048 | 675.6 us | 652.2 us | 83.6 us (compute) | 13% |"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """* `MLA_PROLOG_V3_MM1_MIN_TILE` (default 64) changes how many cores the mm1 GEMM uses
  (64 -> 12 cores for Hcq=768).  Setting it to 48 measured a reproducible but small
  ~1.2% gain at T >= 256 (94.7 vs 96.0 us, 682 vs 689 us); at decode the run-to-run
  noise (+-1.5 us, occasionally +-8 us on a shared card) hides any effect, so the
  upstream default is kept."""
NEW2 = """* `MLA_PROLOG_V3_MM1_MIN_TILE` changes how many cores the mm1 GEMM uses (64 -> 12
  cores for Hcq=768).  The port now defaults to **32** (24 cores at prefill): measured
  93.8 vs 96.0 us at T=256 and 675.6 vs 688.5 us at T=2048 (2 repeats each, stable to
  +-0.2 us), decode unchanged within noise."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme timings refreshed")
