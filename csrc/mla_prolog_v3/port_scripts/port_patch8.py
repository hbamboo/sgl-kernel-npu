#!/usr/bin/env python3
"""Drop the EnableGroupComputeOpt=1 dispatch cases.

Those instantiations mix the MXFP8 (E8M0) dequant scale with the per-channel
(f32) rotary dequant helper and do not compile upstream either; the GLM shapes
used here never request them (tiling reports groupOpt=0).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp")

DROP = [
    """        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 1, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, true, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
""",
    """        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 1, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, true, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
""",
]

with io.open(P, encoding="utf-8") as f:
    s = f.read()
for d in DROP:
    assert s.count(d) == 1, (s.count(d), d[:60])
    s = s.replace(d, "")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("dropped group-compute cases")
