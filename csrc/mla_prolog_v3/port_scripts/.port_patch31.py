#!/usr/bin/env python3
"""Default the mm1 minimum tile to 32 (GLM measured +2% at T>=256, no decode change).

Upstream uses 64, which caps mm1 at hcq/64 = 12 cores for GLM's Hcq=768.  32 lets the
GEMM use 24 cores at prefill; measured 93.8 vs 96.0 us at T=256 and 675.6 vs 688.5 us
at T=2048 (2 repeats each), decode unchanged within noise.  The env knob still wins.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """        // [sgl-kernel-npu port] tunable floor (default 64): how many cores mm1 uses.
        uint32_t mm1MinTile = 64U;"""
NEW = """        // [sgl-kernel-npu port] tunable floor: how many cores mm1 uses.  Default 32
        // (upstream 64 caps GLM's Hcq=768 at 12 cores; 32 uses 24 at prefill and
        // measured ~2% faster there with no decode change).
        uint32_t mm1MinTile = 32U;"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("mm1 min tile default -> 32")
