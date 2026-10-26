#!/usr/bin/env python3
"""mm2 uses only 9 of 32 cores (hardcoded baseN=64), so its per-core work is twice mm1's.

FillMatmul2Tiling picks `baseN = 64` and `mm2BlockNum = 576/64 = 9`; the phase therefore
runs 64 columns per core while mm1 runs 32, and the two phases are sequential - so mm2
dominates.  Choosing the smallest 16-aligned tile that still fits in aicNum blocks
(32 -> 18 blocks) halves mm2's per-core work.  Env gated for the experiment.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """    } else if (aicNum_ >= 9U) { // 9是经验值 原切N: 非 FP8/HIF8 模式走这里
        uint32_t baseN = 64U;
        mm2BlockNum_ = hCkvkrSize / baseN;
        singlecoreHeadSizeCkvKr_ = baseN;"""
NEW = """    } else if (aicNum_ >= 9U) { // 9是经验值 原切N: 非 FP8/HIF8 模式走这里
        uint32_t baseN = 64U;
        // [sgl-kernel-npu port] optional: shrink the per-core N so more cores take part
        // (576 cols / 32 = 18 blocks instead of 9, i.e. half the per-core work).
        if (const char *mm2Env = std::getenv("MLA_PROLOG_V3_MM2_BASEN")) {
            const int envBaseN = std::atoi(mm2Env);
            if (envBaseN > 0 && (hCkvkrSize % static_cast<uint32_t>(envBaseN)) == 0U &&
                (hCkvkrSize / static_cast<uint32_t>(envBaseN)) <= aicNum_) {
                baseN = static_cast<uint32_t>(envBaseN);
            }
        }
        mm2BlockNum_ = hCkvkrSize / baseN;
        singlecoreHeadSizeCkvKr_ = baseN;"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("mm2 baseN knob added")
