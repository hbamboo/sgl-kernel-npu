#!/usr/bin/env python3
"""Raise the per-step token cap for D=192 (GLM) from 64 to 128.

Upstream caps D != 128 at 64 tokens/step for UB-budget reasons.  On Ascend950 with
the GLM geometry the wave buffers fit 128 (validated against the CPU golden at
T=256/512/1024) and it is ~4% faster; 192 tokens/step exceeds the budget and faults,
so the cap stays at 64 for any other D.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """    if (baseShapeInfo_.dSize != HIGH_THROUGHPUT__D_SIZE && waveBufAllocated) {
        maxStepBatch = 64U;
    }"""
NEW = """    if (baseShapeInfo_.dSize != HIGH_THROUGHPUT__D_SIZE && waveBufAllocated) {
        // [sgl-kernel-npu port] upstream uses 64 for every D != 128.  With D=192
        // (GLM-4.7-Flash) 128 tokens/step fits the UB budget on Ascend950 - validated
        // against the CPU golden at T=256/512/1024 and ~4% faster - while 192
        // tokens/step faults, so other D values keep the conservative 64.
        maxStepBatch = (baseShapeInfo_.dSize == 192U) ? 128U : 64U;
    }"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("D=192 step cap -> 128")
