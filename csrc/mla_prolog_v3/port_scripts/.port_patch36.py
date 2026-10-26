#!/usr/bin/env python3
"""README: add the prefill pipe profile and the mm3 baseK finding."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """Two experiments that did **not** pan out (both reverted / kept opt-in):"""
NEW = """Prefill (T=256, 32 cores, median task 94 us) shows the same picture, more strongly:
AIC MTE1 **49%**, MTE2 34%, **MAC only 20%**, scalar 13%.  So even at T=256 the kernel
is L1->L0 movement bound and the cube is starved - the "13% of the compute roofline"
is caused by feeding, not by the GEMM math.  Raising the mm3 baseK (64 -> 128, stepK
compensated) measured no change at all (93.8 / 93.9 / 93.9 us), so mm3's L0 tile shape
is not the limiter; the remaining gap needs a different data-flow (deeper L1 pipeline
or fusing the phases), i.e. kernel work.

Three experiments that did **not** pan out (reverted / kept opt-in):"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme updated")
