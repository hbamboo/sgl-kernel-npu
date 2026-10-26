#!/usr/bin/env python3
"""Replace GET_TILING_DATA_WITH_STRUCT (not provided by the direct-launch build)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

pairs = [
    ("""#include "../../mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h\"""",
     """#include "mla_prolog_v3_tiling_copy.h"
#include "../../mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h\""""),
    ("""    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    const optiling::MlaPrologTilingData* __restrict tilingData = nullptr;""",
     """    // [sgl-kernel-npu port] the direct-launch build has no GET_TILING_DATA_WITH_STRUCT;
    // copy the GM tiling blob into local memory explicitly.
    optiling::MlaPrologTilingData tilingDataIn;
    optiling::CopyMlaPrologTilingDataFromGm(tiling, tilingDataIn);
    const optiling::MlaPrologTilingData* __restrict tilingData = nullptr;"""),
    ("""    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    const uint64_t key = tilingDataIn.tilingKey;""",
     """    const uint64_t key = reinterpret_cast<const __gm__ optiling::MlaPrologTilingData *>(tiling)->tilingKey;"""),
]

for old, new in pairs:
    n = s.count(old)
    if n != 1:
        raise SystemExit("pattern count %d for %r" % (n, old[:60]))
    s = s.replace(old, new)

with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("patched tiling access")
