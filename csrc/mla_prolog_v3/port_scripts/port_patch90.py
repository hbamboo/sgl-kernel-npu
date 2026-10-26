#!/usr/bin/env python3
"""README: add the before/after kernel profile next to the roofline table."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """Steady-state msprof `PipeUtilization` on the decode case"""
NEW = """Steady-state kernel profile, same script and device (msprof `PipeUtilization`, median):

| metric | before (32 cores, first port) | now (16 cores) |
|---|---|---|
| kernel time (T=1) | 28.96 us | **18.64 us (-36%)** |
| AIC MAC | 9.5% | **29.3%** |
| AIC MTE1 | 15.3% | 51.2% |
| AIC MTE2 | 35.9% | 48.9% |
| AIC scalar | 30.7% | 36.2% |

Prefill (T=256, 32 cores): MTE1 **56.4%** (busiest), MTE2 28.4%, MAC 17.9%, scalar 10% -
i.e. the L1->L0 feed, which matches the "L1 fully allocated" finding below.

Steady-state msprof `PipeUtilization` on the decode case"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme: profile table added")
