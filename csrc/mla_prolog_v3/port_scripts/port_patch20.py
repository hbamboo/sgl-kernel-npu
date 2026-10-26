#!/usr/bin/env python3
"""Record the MXFP8 numerics findings (per-path, per-core-count) in the README."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """* [ ] native MXFP8 numerics: first GEMM correct, `weight_uq_qr` dequant explodes for
      `Hcq=768` (24 scale bytes/row, not 32 B aligned) while `Hcq=1536` is fine"""
NEW = """* [ ] native MXFP8 numerics for `Hcq=768` (`test_mxfp8_precision.py`, T=8):

      | path | 16 cores | 32 cores |
      |---|---|---|
      | kv_cache / kr_cache (mm2) | cosine 0.9999989 | cosine 0.9999989 |
      | query / query_rope (mm3+mm4) | cosine 0.981 | garbage (1e37) |

      So the down-projection GEMMs are fine; the `weight_uq_qr` per-32 scale mapping is
      off by ~2% at 16 cores and reads out of bounds at 32 cores (core-split offset
      computation with 24 scale bytes per row).  bf16 (dequantised) weights stay the
      validated path."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme mxfp8 status updated")
