#!/usr/bin/env python3
"""Refresh the performance section with the auto core-count heuristic results."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

OLD_START = "## Performance (Ascend950DT) and roofline"
OLD_END = "## Cross-check against the CANN built-in op"
with io.open(P, encoding="utf-8") as f:
    s = f.read()
i0 = s.index(OLD_START)
i1 = s.index(OLD_END)

NEW = """## Performance (Ascend950DT) and roofline

Measured machine limits on this card:

* HBM bandwidth (256 MiB copy): **3.27 TB/s**
* bf16 matmul (4096^3): **424 TFLOP/s**
* machine balance: 424 / 3.27 = **130 FLOP/byte**
* **framework floor**: any NPU kernel op costs **23.4 us** per call on this stack
  (`helloworld`, tiny `qkv_rms_norm_rope_cache`).  The op itself adds ~0.7 us on top.

Per call the op streams 17.3 MB of bf16 weights and does 17.3 MFLOP per token, so it
is memory bound up to T ~ 130 and compute bound above.

Latest timings (auto core count: 16 cores for T <= 128, 32 above):

| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 41.6 us | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.7 us | 18.3 us | 5.4 us (BW) | 29% |
| 16 | 41.4 us | 18.0 us | 5.6 us (BW) | 31% |
| 64 | 54.3 us | 30.9 us | 7.4 us (BW) | 24% |
| 256 | 96.0 us | 72.6 us | 10.4 us (compute) | 14% |
| 2048 | 688.7 us | 665.3 us | 83.6 us (compute) | 13% |

Progression of the decode path: 77.7 us (first working version) -> 48.5 us (tiling +
workspace cache) -> 45.1 us (hash keys) -> **41.6 us** (core-count heuristic), i.e.
**1.87x** faster than the first working port.

Core-count behaviour (`MLA_PROLOG_V3_CORES` overrides, default is the heuristic):

| cores | T=1 | T=8 | T=16 | T=32 | T=128 | T=256 | T=2048 |
|---|---|---|---|---|---|---|---|
| 32 | 44.6 | 44.5 | 45.4 | 47.5 | 59.4 | 95.8 | 688.7 |
| 16 | 42.1 | 42.6 | 42.1 | 45.1 | 58.2 | 111.7 | 858.5 |
| 8 | 44.4 | 48.3 | - | - | - | 137.1 | 1065 |

msprof `PipeUtilization` on the decode case (T=1, 32 cores) shows where the remaining
time goes: MTE2 (weight DMA) 36%, **AIC scalar 31%**, MTE1 15%, MAC 9.5% - i.e. the
GEMV-shaped GEMMs are limited by scalar bookkeeping and small DMA requests, not by
the cube or by raw bandwidth.

"""
s = s[:i0] + NEW + s[i1:]

s = s.replace("""* [ ] kernel-side performance work: fp8/MXFP8 weight streaming (halves the bytes) or
      a GLM-specific retiling (32 cores leave only 24 columns per core)""",
              """* [x] decode core-count heuristic (16 cores for T <= 128): 45.1 -> 41.6 us
* [ ] kernel-side performance work: the profiler shows 31% AIC scalar + 36% MTE2 at
      decode, so the next step is a GLM-specific GEMV path (larger DMA requests, less
      per-token scalar work) and/or fp8 weight streaming to halve the bytes""")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme perf refreshed")
