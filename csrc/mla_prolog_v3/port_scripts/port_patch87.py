#!/usr/bin/env python3
"""README: record the mm2 core-split negative result (and its implication for fusion)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """**The kernel's L1 is 100% allocated**"""
NEW = """**mm2 core-split does not help either** (measured): `FillMatmul2Tiling` hardcodes
`baseN = 64`, so mm2 runs 9 blocks x 64 columns while mm1 runs 24 x 32 - and since the two
phases are sequential, halving mm2's per-core columns looked like an easy win.  Making it
18 blocks x 32 columns (a legal 16-aligned tile) measured only 89.35 vs 89.75 us at
T=256 and 645.7 vs 649.8 us at T=2048 (<1%), so the phase is *not* limited by per-core
column count.  This also kills the "fuse mm1+mm2 for a bigger N per core" estimate: the
only benefit left there would be removing one phase boundary (~2-4 us of 70), i.e. 3-6%
of prefill - not worth a tiling + kernel-output-split rewrite.

**The kernel's L1 is 100% allocated**"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme: mm2 split result recorded")
