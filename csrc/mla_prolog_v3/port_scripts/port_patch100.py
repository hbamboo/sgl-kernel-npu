"""Route A step 1: tiling support for a fused down-projection weight (corrected v2).

The member lives in mla_prolog_tiling.h; the .cpp only gets the logic.
"""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"

# ---- header: member
PH = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.h"
s = io.open(PH, encoding="utf-8").read()
if "fusedDownN_" not in s:
    old = "    uint32_t mm1BlockNum_ = 0;"
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, old + "\n    uint32_t fusedDownN_ = 0; // [port] >0 when the down projections are fused")
    io.open(PH, "w", encoding="utf-8").write(s)
    print("header: member added")

# ---- cpp: logic
PC = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.cpp"
s = io.open(PC, encoding="utf-8").read()

if s.count("fusedDownN") == 0:
    old = ("    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;\n"
           "    return ge::GRAPH_SUCCESS;\n}")
    new = ("    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;\n"
           "    // [port] fused down projection: `weight_dkv_kr` then holds [He, Hcq+Hckv+Dr], so\n"
           "    // Hckv cannot come from its shape - take it from the KV cache instead.\n"
           "    fusedDownN_ = 0U;\n"
           "    if (const char *fuseEnv = std::getenv(\"MLA_PROLOG_V3_FUSE_DOWN\")) {\n"
           "        if (std::atoi(fuseEnv) == 1) {\n"
           "            const auto &kvShape = context_->kvCache.shape->GetStorageShape();\n"
           "            if (kvShape.GetDimNum() == MLA_PROLOG_DIM_NUM_4) {\n"
           "                baseShapeInfo_.hckvSize = kvShape.GetDim(MLA_PROLOG_DIM_INDEX_3);\n"
           "            }\n"
           "            fusedDownN_ = baseShapeInfo_.hcqSize + baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;\n"
           "        }\n"
           "    }\n"
           "    return ge::GRAPH_SUCCESS;\n}")
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = ("    mm1NGroup_ = 1;\n    mm1KGroup_ = 1;\n    mm1SingleCoreK_ = 0;\n"
           "    if (scenarioInfo_.splitMFlag_ == 1U) {")
    new = ("    mm1NGroup_ = 1;\n    mm1KGroup_ = 1;\n    mm1SingleCoreK_ = 0;\n"
           "    if (fusedDownN_ > 0U) {\n"
           "        // [port] one GEMM covers both down projections\n"
           "        singlecoreHeadSizeCq_ = CalcSingleCoreN(fusedDownN_, aicNum_, 16U);\n"
           "        singlecoreHeadSizeCq_ = std::max(singlecoreHeadSizeCq_, 32U);\n"
           "        mm1BlockNum_ = CeilDiv(fusedDownN_, singlecoreHeadSizeCq_);\n"
           "        return ge::GRAPH_SUCCESS;\n"
           "    }\n"
           "    if (scenarioInfo_.splitMFlag_ == 1U) {")
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = ("ge::graphStatus MlaPrologTiling::CalcWorkSpace()\n{\n"
           "    workspaceSize_ = libapiSize_;")
    new = ("ge::graphStatus MlaPrologTiling::CalcWorkSpace()\n{\n"
           "    workspaceSize_ = libapiSize_;\n"
           "    // [port] fused down projection writes [stepBatch, fusedDownN] into the Cq slab\n"
           "    if (fusedDownN_ > 0U) {\n"
           "        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *\n"
           "                          static_cast<size_t>(fusedDownN_ - baseShapeInfo_.hcqSize) *"
           " static_cast<size_t>(2U);\n"
           "    }")
    assert s.count(old) == 1
    s = s.replace(old, new)

    old = "    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"
    new = ("    baseParams_->fusedDownN = fusedDownN_;\n"
           "    baseParams_->kvCacheStride0 = context_->kvCacheStride0;")
    assert s.count(old) == 1
    s = s.replace(old, new)

    io.open(PC, "w", encoding="utf-8").write(s)
    print("cpp: fused down projection wired")
else:
    print("cpp: already patched")
