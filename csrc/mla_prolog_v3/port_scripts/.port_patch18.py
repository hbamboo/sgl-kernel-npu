#!/usr/bin/env python3
"""Pick the core count from the token count (GLM decode heuristic).

Measured on Ascend950DT (GLM-4.7-Flash geometry, bf16): 16 cores win for
T <= 128 (fewer, larger per-core tiles), 32 cores win above that.  An explicit
MLA_PROLOG_V3_CORES override always takes precedence.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.h")
C = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

with io.open(H, encoding="utf-8") as f:
    s = f.read()
old = """    ge::graphStatus SetShapeInfo();"""
new = """    ge::graphStatus SetShapeInfo();
    // [sgl-kernel-npu port] decode heuristic: fewer cores copy larger tiles.
    ge::graphStatus AdjustCoreNumByShape();"""
assert s.count(old) == 1
with io.open(H, "w", encoding="utf-8") as f:
    f.write(s.replace(old, new))

with io.open(C, encoding="utf-8") as f:
    s = f.read()

impl = """
// [sgl-kernel-npu port] GLM decode heuristic.  With T <= 128 the per-core N slice at
// 32 cores is only 24 columns of Hcq, which wastes the cube and fragments the DMA;
// using 16 cores (twice the tile) measured ~6% faster.  Override wins when set.
ge::graphStatus MlaPrologTiling::AdjustCoreNumByShape()
{
    if (std::getenv("MLA_PROLOG_V3_CORES") != nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    constexpr uint32_t kDecodeTokenLimit = 128U;
    if (baseShapeInfo_.tSize <= kDecodeTokenLimit && aicNum_ > 16U) {
        aicNum_ = 16U;
        aivNum_ = 32U;
    }
    return ge::GRAPH_SUCCESS;
}
"""
marker = "ge::graphStatus MlaPrologTiling::SetShapeInfo()"
idx = s.index(marker)
# insert the new function right before SetShapeInfo's definition
s = s[:idx] + impl.lstrip("\n") + "\n" + s[idx:]

old_list = """        std::bind(&MlaPrologTiling::SetShapeInfo, this),"""
new_list = """        std::bind(&MlaPrologTiling::SetShapeInfo, this),
        std::bind(&MlaPrologTiling::AdjustCoreNumByShape, this),"""
assert s.count(old_list) == 1
s = s.replace(old_list, new_list)
with io.open(C, "w", encoding="utf-8") as f:
    f.write(s)
print("core heuristic added")
