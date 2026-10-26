#!/usr/bin/env python3
"""Knob: minimum mm1 tile (controls how many cores mm1 uses).

`singlecoreHeadSizeCq_ = max(CalcSingleCoreN(hcq, aicNum, 16), 64)` caps mm1 at
hcq/64 cores (12 for GLM's Hcq=768), which leaves most cores idle during that phase.
Make the floor tunable to sweep the trade-off (bigger tiles vs more parallelism).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """        singlecoreHeadSizeCq_ = std::max(singlecoreHeadSizeCq_, 64U); // 64：最大使用24核"""
NEW = """        // [sgl-kernel-npu port] tunable floor (default 64): how many cores mm1 uses.
        uint32_t mm1MinTile = 64U;
        if (const char *tileEnv = std::getenv("MLA_PROLOG_V3_MM1_MIN_TILE")) {
            const int tile = std::atoi(tileEnv);
            if (tile > 0) {
                mm1MinTile = static_cast<uint32_t>(tile);
            }
        }
        singlecoreHeadSizeCq_ = std::max(singlecoreHeadSizeCq_, mm1MinTile); // 64：最大使用24核"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("mm1 min-tile knob added")
