#!/usr/bin/env python3
"""Record the negative host-side result in the README (measured, then reverted)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,"""
NEW = """* **host bookkeeping after the PyBind path** (measured with a wall-clock loop, T=1):
  empty binding call 0.11 us, full call 11.8 us, tiling-only (probe) 8.4 us, i.e.
  ~3.4 us launch + ~8.3 us in our own host code.  Most of that is rebuilding the
  `MlaPrologV3Tensors` description (20 `std::vector` allocations) and the tiling lookup.
  Trying to cache the description and revalidate it allocation-free was **slower**
  (15.3 us vs 11.8 us: the ~25 shape/dtype comparisons cost more than the rebuild), so
  that attempt was reverted - the simple path stays.
* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme: host optimisation negative result recorded")
