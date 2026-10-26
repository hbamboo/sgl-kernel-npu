#!/usr/bin/env python3
"""Port patch #1: make mla_prolog tiling usable from a direct-launch host.

* replace fe::PlatFormInfos* platform queries with PlatformAscendCManager
* record the tiling-key template arguments into the tiling data so the
  single kernel entry can dispatch at runtime (no ASCENDC_TPL machinery).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def patch(rel, pairs):
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8") as f:
        s = f.read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit("pattern count %d in %s for %r" % (n, rel, old[:80]))
        s = s.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("patched", rel)


# ---------------------------------------------------------------- tiling data
patch("csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h", [
    ("""struct MlaPrologTilingData {
    MlaPrologBaseParams baseParams; // \u5d4c\u5957\u57fa\u7840\u53c2\u6570\u7ed3\u6784\u4f53\uff08\u5305\u542b\u7ef4\u5ea6\u3001\u5934\u4fe1\u606f\u7b49\u6838\u5fc3\u914d\u7f6e\uff09
};""",
     """// [sgl-kernel-npu port] tiling-key template arguments, recorded by the host tiling code
// so that the single direct-launch kernel entry can dispatch on them at runtime
// instead of relying on the ASCENDC_TPL binary-selection machinery.
struct MlaPrologTplArgs {
    uint32_t cacheMode;
    uint32_t scenario;
    uint32_t quantMode;
    uint32_t enableDequantOpt;
    uint32_t enableGroupComputeOpt;
    uint32_t emptyTensorMode;
    uint32_t actualSeqLenMode;
    uint32_t splitMMode;
    uint32_t enableRope;
    uint32_t cvMode;
};

struct MlaPrologTilingData {
    MlaPrologBaseParams baseParams; // \u5d4c\u5957\u57fa\u7840\u53c2\u6570\u7ed3\u6784\u4f53\uff08\u5305\u542b\u7ef4\u5ea6\u3001\u5934\u4fe1\u606f\u7b49\u6838\u5fc3\u914d\u7f6e\uff09
    MlaPrologTplArgs tplArgs;       // [sgl-kernel-npu port] runtime template-argument record
};"""),
])

# ---------------------------------------------------------------- context
patch("csrc/mla_prolog/op_host/mla_prolog_tiling.h", [
    ("""    size_t* workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};""",
     """    size_t* workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
    // [sgl-kernel-npu port] decoded tiling-key template arguments (filled by GenTilingKey)
    MlaPrologTplArgs tplArgs{};
};"""),
])

# ---------------------------------------------------------------- platform queries
patch("csrc/mla_prolog/op_host/mla_prolog_tiling.cpp", [
    ("""NpuArch MlaPrologTiling::GetCurNpuArch() const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);""",
     """NpuArch MlaPrologTiling::GetCurNpuArch() const
{
    // [sgl-kernel-npu port] direct-launch host side has no fe::PlatFormInfos*; use the
    // global platform singleton (initialised by aclInit inside the torch extension).
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();"""),
    ("""ge::graphStatus MlaPrologTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);""",
     """ge::graphStatus MlaPrologTiling::GetNpuInfo()
{
    // [sgl-kernel-npu port] platform singleton instead of fe::PlatFormInfos*.
    auto ascendcPlatformPtr = platform_ascendc::PlatformAscendCManager::GetInstance();
    OP_CHECK_IF(ascendcPlatformPtr == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "PlatformAscendCManager is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = *ascendcPlatformPtr;"""),
    # record template args (empty-tensor branch)
    ("""    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        context_->tilingKey = GET_TPL_TILING_KEY(
            0, 0, 0, false, false, static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_), 0, 0, true, cvMode);""",
     """    // [sgl-kernel-npu port] record the template arguments for the runtime-dispatch entry
    context_->tplArgs.cacheMode = 0U;
    context_->tplArgs.scenario = 0U;
    context_->tplArgs.quantMode = 0U;
    context_->tplArgs.enableDequantOpt = 0U;
    context_->tplArgs.enableGroupComputeOpt = 0U;
    context_->tplArgs.emptyTensorMode = static_cast<uint32_t>(scenarioInfo_.emptyTensorMode_);
    context_->tplArgs.actualSeqLenMode = 0U;
    context_->tplArgs.splitMMode = 0U;
    context_->tplArgs.enableRope = 1U;
    context_->tplArgs.cvMode = cvMode;

    if (scenarioInfo_.emptyTensorMode_ == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        context_->tilingKey = GET_TPL_TILING_KEY(
            0, 0, 0, false, false, static_cast<uint8_t>(scenarioInfo_.emptyTensorMode_), 0, 0, true, cvMode);"""),
    ("""        uint8_t cacheMode =
            scenarioInfo_.cacheMode_ == CACHE_MODE::TND ? 0 : static_cast<uint8_t>(scenarioInfo_.cacheMode_);
        context_->tilingKey = GET_TPL_TILING_KEY(""",
     """        uint8_t cacheMode =
            scenarioInfo_.cacheMode_ == CACHE_MODE::TND ? 0 : static_cast<uint8_t>(scenarioInfo_.cacheMode_);
        // [sgl-kernel-npu port] record the template arguments for the runtime-dispatch entry
        context_->tplArgs.cacheMode = cacheMode;
        context_->tplArgs.scenario = typeValue;
        context_->tplArgs.quantMode = quantType;
        context_->tplArgs.enableDequantOpt = enableDequantOpt_ ? 1U : 0U;
        context_->tplArgs.enableGroupComputeOpt = enableGroupComputeOpt_ ? 1U : 0U;
        context_->tplArgs.emptyTensorMode = static_cast<uint32_t>(scenarioInfo_.emptyTensorMode_);
        context_->tplArgs.actualSeqLenMode = static_cast<uint32_t>(scenarioInfo_.actualSeqMode_);
        context_->tplArgs.splitMMode = static_cast<uint32_t>(scenarioInfo_.splitMFlag_);
        context_->tplArgs.enableRope = enableRope_ ? 1U : 0U;
        context_->tplArgs.cvMode = cvMode;
        context_->tilingKey = GET_TPL_TILING_KEY("""),
])

patch("csrc/mla_prolog/op_host/mla_prolog_tiling_check.cpp", [
    ("""NpuArch MlaPrologTilingCheck::GetCurNpuArch() const
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_.platformInfo);""",
     """NpuArch MlaPrologTilingCheck::GetCurNpuArch() const
{
    // [sgl-kernel-npu port] platform singleton instead of fe::PlatFormInfos*.
    auto ascendcPlatform = *platform_ascendc::PlatformAscendCManager::GetInstance();"""),
])

print("done")
