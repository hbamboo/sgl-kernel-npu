#!/usr/bin/env python3
"""README: record the L1-capacity proof of the kernel's ceiling."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """**Why the L0/L1 tile cannot simply be made larger** (probe, hardcoded values):"""
NEW = """**The kernel's L1 is 100% allocated** (`mla_prolog_comm_arch35.h`):

| constant | value | meaning |
|---|---|---|
| `L1_A_SIZE` | 128 KB, used as `* 2` (ping/pong) | A staging = 256 KB |
| `L1_B_SIZE` | 128 KB, used as `* 2` | B staging = 256 KB |
| total | **512 KB** | the whole L1 |

and the staging per iteration is `kL1StepSize = baseK * stepK = 128 * 4 = 512`, which is
*simultaneously* the B limit (128 cols x 512 x 2 B = 128 KB) and - at m = 128 - the A
limit (128 rows x 512 x 2 B = 128 KB).  So there is no headroom in either direction: a
larger baseK/stepK overflows (which is exactly the 507015 fault at baseK=256), and a
smaller baseK only makes the L0 tile shallower.  Decode does not help either: at m = 1
the A buffer is nearly idle but B still needs the full 128 KB, and `kL1StepSize` cannot
grow without overflowing B.

That is the quantitative reason the cube sits at ~7% of peak on these shapes: the kernel
streams weights through a saturated 512 KB L1 with 128 KB ping-pong buffers, so its
throughput is set by L1->L0 transaction granularity, not by anything tunable.

**Why the L0/L1 tile cannot simply be made larger** (probe, hardcoded values):"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme: L1-capacity proof recorded")
