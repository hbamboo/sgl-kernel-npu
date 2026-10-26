#!/usr/bin/env python3
"""Record the aclgraph finding in README-port.md."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """msprof `PipeUtilization` on the decode case"""
NEW = """aclgraph (`torch.npu.NPUGraph`) capture works and replays bit-exactly, but a
single-op graph is **slower** than eager on this stack (150 us replay vs 42 us eager:
graph launch overhead dominates for a one-kernel graph), so graph capture is only
useful for the whole-model graph SGLang builds, not per op.

msprof `PipeUtilization` on the decode case"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme graph note added")
