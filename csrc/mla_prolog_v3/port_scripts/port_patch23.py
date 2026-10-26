#!/usr/bin/env python3
"""Add MLA_PROLOG_V3_MAX_STEP to experiment with the token step size.

The tiling caps the per-step token count at 64 for D != 128 (UB budget), which
means T=256 pays the four-phase fixed cost four times.  Expose the cap so the
step size can be swept (correctness must be re-verified at each setting).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """    stepBatchSize_ = std::min(maxStepBatch, baseShapeInfo_.tSize);"""
NEW = """    // [sgl-kernel-npu port] tuning knob: per-step token cap (UB-budget bound).
    if (const char *stepEnv = std::getenv("MLA_PROLOG_V3_MAX_STEP")) {
        const int step = std::atoi(stepEnv);
        if (step > 0) {
            maxStepBatch = static_cast<uint32_t>(step);
        }
    }
    stepBatchSize_ = std::min(maxStepBatch, baseShapeInfo_.tSize);"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("max-step knob added")
