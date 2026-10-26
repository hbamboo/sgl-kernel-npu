#!/usr/bin/env python3
"""Record the cross-check against the CANN built-in op in README-port.md."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

OLD = """## Status"""
NEW = """## Cross-check against the CANN built-in op

On the DeepSeek-like geometry the built-in binary supports
(`He=7168, Hcq=1536, Hckv=512, D=128, Dr=64, N=16`, bf16), the ported kernel is
**bit-identical** to `torch_npu.npu_mla_prolog_v3` (max abs diff 0.00000 on all five
outputs, `cmp_builtin_numeric.py`) and **15-17% faster** end-to-end because it skips
the aclnn/op-plugin dispatch:

| T | ported | CANN built-in |
|---|---|---|
| 1 | 56.7 us | 66.6 us |
| 8 | 56.1 us | 67.9 us |
| 64 | 66.8 us | 78.4 us |

This also shows that the ~800 GB/s the kernel achieves is inherent to the op (not a
porting regression); improving it needs kernel work.

## Status"""
with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW, 1)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme cross-check added")
