#!/usr/bin/env python3
"""README: record the L1-capacity limit probe (baseK > 128 faults) and correct the
earlier 'baseK neutral' claim."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """Three experiments that did **not** pan out (reverted / kept opt-in):"""
NEW = """**Why the L0/L1 tile cannot simply be made larger** (probe, hardcoded values):

| mm1/mm2 baseK (bf16) | result |
|---|---|
| 128 (upstream) | works, 90.0 us @T=256 |
| 256, stepK=2 | faults (`507015`) once the step is 128 tokens (m=128) |
| 512, stepK=1 | faults immediately |

The kernel sizes its L1 staging as `128KB / (128 baseN * 4 stepK * sizeof(type))`, i.e.
it already sits at the L1 capacity limit; with m=128 the A tile adds another
`baseK*m*2` bytes, so 256/512 overflow.  (This also invalidates the earlier
"baseK 64/128 measured neutral" note: that probe went through an env knob that did not
actually reach the kernel - a hardcoded 256 faults, so it can never have been applied.)

Consequence for the remaining gap: with L1 full and 24 cores on mm1 (n=32 per core),
each core streams only ~131 KB per step yet the phase takes ~24 us, i.e. the cost is
per-transfer / scalar overhead, not bandwidth - and the tile cannot grow without
merging mm1+mm2 (n would become 56/core) or restructuring the phases.  Both are
kernel rewrites, not knobs.

Three experiments that did **not** pan out (reverted / kept opt-in):"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme updated with the L1 probe")
