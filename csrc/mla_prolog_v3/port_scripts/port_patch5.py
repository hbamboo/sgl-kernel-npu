#!/usr/bin/env python3
"""Port patch #5: single named kernel entry with runtime dispatch on tiling key.

`mla_prolog_v3` is normally instantiated by the ASCENDC_TPL machinery (one binary
per template combination).  sgl-kernel-npu launches kernels directly, so the
templated body is kept as an inline function and a single `extern "C"` entry
dispatches on the tiling-key recorded in the tiling data (same trick as the
`compressor` op).
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
            raise SystemExit("pattern count %d in %s for %r" % (n, rel, old[:70]))
        s = s.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("patched", rel)


# ---------------------------------------------------------------- tiling data
patch("csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h", [
    ("""    MlaPrologTplArgs tplArgs;       // [sgl-kernel-npu port] runtime template-argument record
};""",
     """    MlaPrologTplArgs tplArgs;       // [sgl-kernel-npu port] runtime template-argument record
    uint64_t tilingKey;             // [sgl-kernel-npu port] full tiling key (runtime dispatch)
};"""),
])

# ---------------------------------------------------------------- host
patch("csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp", [
    ("""    tilingData->tplArgs = ctx.tplArgs;
    return true;""",
     """    tilingData->tplArgs = ctx.tplArgs;
    tilingData->tilingKey = ctx.tilingKey;
    return true;"""),
])

# ---------------------------------------------------------------- kernel entry
BODY_SIG_OLD = """template <uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode, bool EnableDequantOpt, bool EnableGroupComputeOpt,
          uint8_t EmptyTensorMode, uint8_t ActualSeqLenMode, uint8_t SplitMMode, bool EnableRope, uint8_t CvMode>
__global__ __aicore__ void mla_prolog_v3("""
BODY_SIG_NEW = """template <uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode, bool EnableDequantOpt, bool EnableGroupComputeOpt,
          uint8_t EmptyTensorMode, uint8_t ActualSeqLenMode, uint8_t SplitMMode, bool EnableRope, uint8_t CvMode>
__aicore__ inline void MlaPrologV3KernelBody("""

ENTRY = r"""
// ---------------------------------------------------------------------------
// [sgl-kernel-npu port] single named kernel entry, dispatch on the tiling key
// that the host tiling recorded in the tiling data.
//
// key layout (see csrc/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h and
// AscendC::FastEncodeTilingKeyDirect):
//   bit 0-3   CACHE_MODE          1=PA_BSND 2=PA_NZ 0=BSND 3/4=PA_BLK
//   bit 4-5   SCENARIO            1=BF16 no-quant 2=quant
//   bit 6-9   QUANT_MODE          QUANT_MODE enum value
//   bit 10    ENABLE_DEQUANT_OPT
//   bit 11    ENABLE_GROUP_COMPUTE_OPT
//   bit 12-13 EMPTY_TENSOR_MODE
//   bit 14-15 ACTUAL_SEQ_LEN_MODE
//   bit 16-17 SPLIT_M_MODE
//   bit 18    ENABLE_ROPE
//   bit 19-26 CV_MODE             7 = MIX_AIC_1_2 (1:2), 6 = MIX_AIC_1_1 (1:1)
// ---------------------------------------------------------------------------
#define MLA_PROLOG_V3_KEY(cacheMode, scenario, quantMode, dequantOpt, groupOpt, emptyMode, seqLenMode, splitMMode,   \
                          enableRope, cvMode)                                                                       \
    ((static_cast<uint64_t>(cacheMode)) | (static_cast<uint64_t>(scenario) << 4) |                                  \
     (static_cast<uint64_t>(quantMode) << 6) | (static_cast<uint64_t>(dequantOpt) << 10) |                          \
     (static_cast<uint64_t>(groupOpt) << 11) | (static_cast<uint64_t>(emptyMode) << 12) |                           \
     (static_cast<uint64_t>(seqLenMode) << 14) | (static_cast<uint64_t>(splitMMode) << 16) |                        \
     (static_cast<uint64_t>(enableRope) << 18) | (static_cast<uint64_t>(cvMode) << 19))

#define MLA_PROLOG_V3_ARGS                                                                                          \
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,          \
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,         \
        quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut,          \
        kvCacheOut, krCacheOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace

extern "C" __global__ __aicore__ void mla_prolog_v3(
    __gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr, __gm__ uint8_t* weightUk,
    __gm__ uint8_t* weightDkvKr, __gm__ uint8_t* rmsnormGammaCq, __gm__ uint8_t* rmsnormGammaCkv,
    __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos, __gm__ uint8_t* kvCache, __gm__ uint8_t* krCache,
    __gm__ uint8_t* cacheIndex, __gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq,
    __gm__ uint8_t* dequantScaleWUqQr, __gm__ uint8_t* dequantScaleWDkvKr, __gm__ uint8_t* quantScaleCkv,
    __gm__ uint8_t* quantScaleCkr, __gm__ uint8_t* smoothScalesCq, __gm__ uint8_t* actualSeqLen,
    __gm__ uint8_t* kNopeClipAlpha, __gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut, __gm__ uint8_t* kvCacheOut,
    __gm__ uint8_t* krCacheOut, __gm__ uint8_t* dequantScaleQNopeOut, __gm__ uint8_t* queryNormOut,
    __gm__ uint8_t* dequantScaleQNormOut, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    const uint64_t key = tilingDataIn.tilingKey;

    switch (key) {
        // ---- bf16 / no quant, PA_BSND, 1:2 (golden / accuracy baseline)
        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 1, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, true, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;

        // ---- MXFP8 weights (GLM-4.7-Flash), bf16 kv cache, PA_BSND, 1:2
        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 0, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, false, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 0, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, false, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 1, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, true, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        default:
            // Unsupported combination for this build: nothing is written.
            // The host side checks the same key set before launching.
            break;
    }
}
"""

patch("csrc/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp", [
    (BODY_SIG_OLD, BODY_SIG_NEW),
])
with io.open(os.path.join(ROOT, "csrc/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp"), "a", encoding="utf-8") as f:
    f.write(ENTRY)
print("appended entry")
print("done")
