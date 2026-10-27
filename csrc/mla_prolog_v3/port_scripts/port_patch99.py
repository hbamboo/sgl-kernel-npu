"""Route A, step 1/3: tiling support for a fused down-projection weight.

When MLA_PROLOG_V3_FUSE_DOWN=1 the host passes one [He, Hcq+Hckv+Dr] weight in the
`weight_dkv_kr` slot (mm2 is unused then) and the tiling has to:

  * take Hckv from the kv cache instead of the (now fused) weight shape,
  * split mm1 over the fused N so one GEMM covers both down projections,
  * reserve the wider [stepBatch, fusedN] output slab in the workspace.

Everything is env gated; with the flag off the tiling is byte-for-byte the previous one.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1) tiling data: one new field
P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD1 = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements"""
NEW1 = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements
    uint32_t fusedDownN;     // [port] Hcq+Hckv+Dr when the down projections are fused, else 0"""
assert s.count(OLD1) == 1
s = s.replace(OLD1, NEW1)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

# 2) tiling: member, shape override, mm1 split, workspace
P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()

OLD2 = """    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;
    return ge::GRAPH_SUCCESS;
}"""
NEW2 = """    baseShapeInfo_.s2Size = baseShapeInfo_.nkvSize;
    // [port] fused down-projection: `weight_dkv_kr` then holds [He, Hcq+Hckv+Dr], so Hckv
    // cannot be derived from its shape - take it from the KV cache instead.
    fusedDownN_ = 0U;
    if (const char *fuseEnv = std::getenv("MLA_PROLOG_V3_FUSE_DOWN")) {
        if (std::atoi(fuseEnv) == 1) {
            const auto &kvShape = context_->kvCache.shape->GetStorageShape();
            if (kvShape.GetDimNum() == MLA_PROLOG_DIM_NUM_4) {
                baseShapeInfo_.hckvSize = kvShape.GetDim(MLA_PROLOG_DIM_INDEX_3);
            }
            fusedDownN_ = baseShapeInfo_.hcqSize + baseShapeInfo_.hckvSize + baseShapeInfo_.drSize;
        }
    }
    return ge::GRAPH_SUCCESS;
}"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = """    mm1NGroup_ = 1;
    mm1KGroup_ = 1;
    mm1SingleCoreK_ = 0;
    if (scenarioInfo_.splitMFlag_ == 1U) {"""
NEW3 = """    mm1NGroup_ = 1;
    mm1KGroup_ = 1;
    mm1SingleCoreK_ = 0;
    if (fusedDownN_ > 0U) {
        // [port] one GEMM covers both down projections
        singlecoreHeadSizeCq_ = CalcSingleCoreN(fusedDownN_, aicNum_, 16U);
        singlecoreHeadSizeCq_ = std::max(singlecoreHeadSizeCq_, 32U);
        mm1BlockNum_ = CeilDiv(fusedDownN_, singlecoreHeadSizeCq_);
        return ge::GRAPH_SUCCESS;
    }
    if (scenarioInfo_.splitMFlag_ == 1U) {"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)

OLD4 = """ge::graphStatus MlaPrologTiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;"""
NEW4 = """ge::graphStatus MlaPrologTiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;
    // [port] fused down-projection writes [stepBatch, fusedDownN] into the Cq slab
    if (fusedDownN_ > 0U) {
        workspaceSize_ += static_cast<size_t>(stepBatchSize_) *
                          static_cast<size_t>(fusedDownN_ - baseShapeInfo_.hcqSize) * static_cast<size_t>(2U);
    }"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)

OLD5 = """    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
NEW5 = """    baseParams_->fusedDownN = fusedDownN_;
    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
assert s.count(OLD5) == 1
s = s.replace(OLD5, NEW5)

# member declaration
OLD6 = """    uint32_t mm3BlockNum_{0};"""
if OLD6 in s:
    s = s.replace(OLD6, OLD6 + "\n    uint32_t fusedDownN_{0}; // [port] >0 when the down projections are fused")
else:
    # fall back: put it next to the first BlockNum member
    import re
    m = re.search(r"(\n\s*uint32_t mm1BlockNum_\{0\};)", s)
    assert m, "cannot find a member to anchor the new field"
    s = s[:m.end(1)] + "\n    uint32_t fusedDownN_{0}; // [port] >0 when the down projections are fused" + s[m.end(1):]
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("tiling: fused down projection wired")
