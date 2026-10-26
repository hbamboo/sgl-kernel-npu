#!/usr/bin/env python3
"""Experiment: sweep the mm1/mm2 baseK (L0 tile depth) for the GLM geometry.

mm1/mm2 have K=2048 and only 32-64 output columns per core, so the default
baseK=128/stepK=4 issues 16 small L0 loads.  A larger baseK (same L1 footprint via
stepK compensation) means fewer, larger L0 transactions - the profile says AIC MTE1
is the busiest pipe (47-49%).  Env gated, default unchanged.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1) tiling data field
P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements"""
NEW = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements
    uint32_t mm12BaseK;      // [sgl-kernel-npu port] 0 = kernel default (L0 depth sweep)"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

# 2) tiling: env -> baseParams
P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
NEW2 = """    baseParams_->mm12BaseK = 0U;
    if (const char *mmBaseKEnv = std::getenv("MLA_PROLOG_V3_MM_BASEK")) {
        const int baseK = std::atoi(mmBaseKEnv);
        if (baseK > 0) {
            baseParams_->mm12BaseK = static_cast<uint32_t>(baseK);
        }
    }
    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)

# 3) kernel: honour it in mm1 and mm2
P3 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h")
with io.open(P3, encoding="utf-8") as f:
    s = f.read()

HELPERS = """
// [sgl-kernel-npu port] optional baseK override shared by mm1/mm2 (env gated).  The
// L1 footprint (baseK * stepK) is kept at the default 128 * 4 so only the L0 tile
// shape changes; stepK is clamped to at least 1.
template <typename P>
__aicore__ inline void ApplyBaseKOverride(P &param, uint32_t baseKOverride)
{
    if (baseKOverride == 128U || baseKOverride == 256U || baseKOverride == 512U) {
        param.baseK = baseKOverride;
        const uint32_t stepK = (128U * 4U) / baseKOverride;
        param.stepK = static_cast<decltype(param.stepK)>(stepK == 0U ? 1U : stepK);
    }
}
"""
anchor = "template <typename MLAPT>\n__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmCqParamInit()"
assert s.count(anchor) == 1
s = s.replace(anchor, HELPERS + "\n" + anchor)

OLD3 = """    mmCqParam_.stepK = 4;
    if ((mmCqParam_.k / mmCqParam_.baseK) % mmCqParam_.stepK != 0) {
        mmCqParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCqParam_.kScale = mmCqParam_.k / FP8_E4M3_BLOCK_SIZE;"""
NEW3 = """    mmCqParam_.stepK = 4;
    if ((mmCqParam_.k / mmCqParam_.baseK) % mmCqParam_.stepK != 0) {
        mmCqParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    ApplyBaseKOverride(mmCqParam_, baseParams_->mm12BaseK);
    mmCqParam_.kScale = mmCqParam_.k / FP8_E4M3_BLOCK_SIZE;"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)

OLD4 = """    mmCkvKrParam_.stepK = 4;
    if ((mmCkvKrParam_.k / mmCkvKrParam_.baseK) % mmCkvKrParam_.stepK != 0) {
        mmCkvKrParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;"""
NEW4 = """    mmCkvKrParam_.stepK = 4;
    if ((mmCkvKrParam_.k / mmCkvKrParam_.baseK) % mmCkvKrParam_.stepK != 0) {
        mmCkvKrParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    ApplyBaseKOverride(mmCkvKrParam_, baseParams_->mm12BaseK);
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P3, "w", encoding="utf-8") as f:
    f.write(s)
print("mm1/mm2 baseK sweep plumbed")
