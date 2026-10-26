#!/usr/bin/env python3
"""Record the fixed-cost vs bandwidth decomposition of the decode kernel."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """aclgraph (`torch.npu.NPUGraph`) capture works"""
NEW = """### Where the decode time actually goes

Sweeping the head count N (which scales the q_b / w_kc weight bytes) at T=1:

| N | weights | total | kernel | implied GB/s |
|---|---|---|---|---|
| 4 | 7.5 MB | 41.9 us | 18.5 us | 425 |
| 20 | 16.5 MB | 42.7 us | 19.3 us | 896 |
| 62 | 40.1 MB | 50.0 us | 26.6 us | 1580 |
| 124 | 75.0 MB | 59.6 us | 36.2 us | 2176 |

Fitting kernel time against bytes gives

```
t_kernel ≈ 16.5 us + 0.26 us/MB      (≈ 3.8 TB/s marginal, L2-assisted)
```

so for the real 16.5 MB of GLM prolog weights **~85% of the decode kernel time is
fixed overhead** (scalar bookkeeping, four serialised GEMM phase setups, cross-core
syncs) and only ~15-25% is weight streaming.  Consequences:

* MXFP8 weights (8.25 MB) would only save ~2 us at decode - not worth the numerics
  risk unless the fixed cost is removed first;
* the next real win has to come from the kernel structure (fewer phases / less
  per-phase scalar work), not from shrinking the weights.

aclgraph (`torch.npu.NPUGraph`) capture works"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme decomposition added")
