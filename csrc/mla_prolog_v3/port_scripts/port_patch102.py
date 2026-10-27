"""Route A-lite v2: concurrent mm1/mm2 on disjoint core ranges (env gated)."""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"
PC = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.cpp"
PH = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.h"

s = io.open(PC, encoding="utf-8").read()

if "MLA_PROLOG_V3_PARALLEL_DOWN" not in s:
    # --- mm1 split (unique anchor: the assignment right before the fused branch)
    old = ("    mm1NGroup_ = 1;\n    mm1KGroup_ = 1;\n    mm1SingleCoreK_ = 0;\n"
           "    if (fusedDownN_ > 0U) {")
    assert s.count(old) == 1, s.count(old)
    new = ("    mm1NGroup_ = 1;\n    mm1KGroup_ = 1;\n    mm1SingleCoreK_ = 0;\n"
           "    if (const char *parEnv = std::getenv(\"MLA_PROLOG_V3_PARALLEL_DOWN\")) {\n"
           "        if (std::atoi(parEnv) == 1) {\n"
           "            // [port] mm1 on [0, b1), mm2 on [b1, b1+b2): both 16-aligned, b1+b2 <= aicNum\n"
           "            const uint32_t hCkvkrSize = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;\n"
           "            uint32_t n1 = CeilDiv(CeilDiv(baseShapeInfo_.hcqSize, aicNum_ / 2U), 16U) * 16U;\n"
           "            uint32_t n2 = CeilDiv(CeilDiv(hCkvkrSize, aicNum_ / 2U - 4U), 16U) * 16U;\n"
           "            const uint32_t b1 = CeilDiv(baseShapeInfo_.hcqSize, n1);\n"
           "            const uint32_t b2 = CeilDiv(hCkvkrSize, n2);\n"
           "            if (b1 + b2 <= aicNum_) {\n"
           "                singlecoreHeadSizeCq_ = n1;\n"
           "                mm1BlockNum_ = b1;\n"
           "                mm2SingleCoreNOverride_ = n2;\n"
           "                mm2BlockNumOverride_ = b2;\n"
           "                mm2BlockStart_ = b1;\n"
           "                return ge::GRAPH_SUCCESS;\n"
           "            }\n"
           "        }\n"
           "    }\n"
           "    if (fusedDownN_ > 0U) {")
    s = s.replace(old, new)

    # --- mm2 split: honour the override computed in FillMatmul1Tiling
    old2 = ("    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {\n"
            "        return ge::GRAPH_SUCCESS;\n    }\n"
            "    uint32_t hCkvkrSize = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;")
    assert s.count(old2) == 1, s.count(old2)
    new2 = ("    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {\n"
            "        return ge::GRAPH_SUCCESS;\n    }\n"
            "    if (mm2BlockNumOverride_ > 0U) {\n"
            "        // [port] concurrent schedule chosen in FillMatmul1Tiling\n"
            "        singlecoreHeadSizeCkvKr_ = mm2SingleCoreNOverride_;\n"
            "        mm2BlockNum_ = mm2BlockNumOverride_;\n"
            "        return ge::GRAPH_SUCCESS;\n"
            "    }\n"
            "    uint32_t hCkvkrSize = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;")
    s = s.replace(old2, new2)

    old3 = "    baseParams_->fusedDownN = fusedDownN_;"
    assert s.count(old3) == 1
    s = s.replace(old3, old3 + "\n    baseParams_->mm2BlockStart = mm2BlockStart_;")
    io.open(PC, "w", encoding="utf-8").write(s)
    print("cpp: concurrent schedule wired")
else:
    print("cpp: already patched")

s = io.open(PH, encoding="utf-8").read()
if "mm2BlockStart_" not in s:
    old = "    uint32_t fusedDownN_ = 0; // [port] >0 when the down projections are fused"
    assert s.count(old) == 1
    s = s.replace(old, old +
                  "\n    uint32_t mm2BlockStart_ = 0; // [port] first cube block of mm2 in the concurrent schedule"
                  "\n    uint32_t mm2SingleCoreNOverride_ = 0; // [port]"
                  "\n    uint32_t mm2BlockNumOverride_ = 0; // [port]")
    io.open(PH, "w", encoding="utf-8").write(s)
    print("header: members added")
