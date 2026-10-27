"""Route A-lite: run the two down projections *concurrently* on disjoint core ranges.

Instead of fusing them into one GEMM (which needs the vector-side strides rewritten), give
mm2 its own block range above mm1's: the two GEMMs then run at the same time on different
cores, so the phase costs max(mm1, mm2) instead of mm1 + mm2 without touching the workspace
layout, the vector reads or the flag protocol (the FINISH_MM_CQ barrier still waits for
every block, and the mm2 blocks - which produce no Cq - merely reach it earlier).

Env gated: MLA_PROLOG_V3_PARALLEL_DOWN=1.
"""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"

# ---- 1) tiling data: where mm2's block range starts
P1 = ROOT + "/csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h"
s = io.open(P1, encoding="utf-8").read()
if "mm2BlockStart" not in s:
    old = "    uint32_t fusedDownN;     // [port] Hcq+Hckv+Dr when the down projections are fused, else 0"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    uint32_t mm2BlockStart;  // [port] first cube block of mm2 when the two run concurrently")
    io.open(P1, "w", encoding="utf-8").write(s)
    print("data: mm2BlockStart added")

# ---- 2) tiling: split the cube blocks between mm1 and mm2
P2 = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.cpp"
s = io.open(P2, encoding="utf-8").read()
if "mm2BlockStart_" not in s:
    old = "    uint32_t mm2BlockNum_ = 0;"
    if s.count(old) != 1:
        old = None
    # member lives in the header; only logic goes here
    old_logic = """    mm2NGroup_ = 1;
    mm2KGroup_ = 1;
    mm2SingleCoreK_ = 0;
    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return ge::GRAPH_SUCCESS;
    }"""
    new_logic = """    mm2NGroup_ = 1;
    mm2KGroup_ = 1;
    mm2SingleCoreK_ = 0;
    mm2BlockStart_ = 0U;
    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return ge::GRAPH_SUCCESS;
    }
    if (const char *parEnv = std::getenv("MLA_PROLOG_V3_PARALLEL_DOWN")) {
        if (std::atoi(parEnv) == 1) {
            // [port] mm1 on [0, b1), mm2 on [b1, b1 + b2): both 16-aligned, together <= aicNum
            const uint32_t hCkvkrSize = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;
            uint32_t b1 = aicNum_ / 2U;
            uint32_t b2 = aicNum_ / 2U - 4U;
            if (b2 == 0U) {
                b2 = 1U;
            }
            uint32_t n1 = CeilDiv(CeilDiv(baseShapeInfo_.hcqSize, b1), 16U) * 16U;
            uint32_t n2 = CeilDiv(CeilDiv(hCkvkrSize, b2), 16U) * 16U;
            b1 = CeilDiv(baseShapeInfo_.hcqSize, n1);
            b2 = CeilDiv(hCkvkrSize, n2);
            if (b1 + b2 <= aicNum_) {
                mm1SingleCoreNOverride_ = n1;
                mm1BlockNum_ = b1;
                singlecoreHeadSizeCkvKr_ = n2;
                mm2BlockNum_ = b2;
                mm2BlockStart_ = b1;
                return ge::GRAPH_SUCCESS;
            }
        }
    }"""
    assert s.count(old_logic) == 1
    s = s.replace(old_logic, new_logic)

    # mm1 must honour the override computed in FillMatmul2Tiling (order: mm1 first!), so
    # recompute the mm1 split here as well - FillMatmul1Tiling runs before mm2's.
    old_mm1 = """    if (fusedDownN_ > 0U) {"""
    new_mm1 = """    if (const char *parEnv0 = std::getenv("MLA_PROLOG_V3_PARALLEL_DOWN")) {
        if (std::atoi(parEnv0) == 1) {
            const uint32_t hCkvkrSize = baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;
            uint32_t b1 = aicNum_ / 2U;
            uint32_t b2 = aicNum_ / 2U - 4U;
            if (b2 == 0U) {
                b2 = 1U;
            }
            uint32_t n1 = CeilDiv(CeilDiv(baseShapeInfo_.hcqSize, b1), 16U) * 16U;
            uint32_t n2 = CeilDiv(CeilDiv(hCkvkrSize, b2), 16U) * 16U;
            b1 = CeilDiv(baseShapeInfo_.hcqSize, n1);
            b2 = CeilDiv(hCkvkrSize, n2);
            if (b1 + b2 <= aicNum_) {
                singlecoreHeadSizeCq_ = n1;
                mm1BlockNum_ = b1;
                return ge::GRAPH_SUCCESS;
            }
        }
    }
    if (fusedDownN_ > 0U) {"""
    assert s.count(old_mm1) == 1
    s = s.replace(old_mm1, new_mm1)

    old_kv = "    baseParams_->fusedDownN = fusedDownN_;"
    new_kv = "    baseParams_->fusedDownN = fusedDownN_;\n    baseParams_->mm2BlockStart = mm2BlockStart_;"
    assert s.count(old_kv) == 1
    s = s.replace(old_kv, new_kv)
    io.open(P2, "w", encoding="utf-8").write(s)
    print("cpp: parallel down-projection tiling wired")
else:
    print("cpp: already patched")

# ---- 3) header: members
PH = ROOT + "/csrc/mla_prolog/op_host/mla_prolog_tiling.h"
s = io.open(PH, encoding="utf-8").read()
if "mm2BlockStart_" not in s:
    old = "    uint32_t fusedDownN_ = 0; // [port] >0 when the down projections are fused"
    assert s.count(old) == 1
    s = s.replace(old, old + "\n    uint32_t mm2BlockStart_ = 0; // [port] first cube block of mm2 in the concurrent schedule\n"
                             "    uint32_t mm1SingleCoreNOverride_ = 0; // [port] unused placeholder")
    io.open(PH, "w", encoding="utf-8").write(s)
    print("header: members added")
