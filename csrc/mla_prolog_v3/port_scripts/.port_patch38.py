#!/usr/bin/env python3
"""Revert the mm1/mm2 baseK sweep: measured neutral (41.9-43.3 us vs 41.7-50.2 us
default, i.e. inside the run-to-run noise) and it only added a dead knob."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""
    uint32_t mm12BaseK;      // [sgl-kernel-npu port] 0 = kernel default (L0 depth sweep)""", "")
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    baseParams_->mm12BaseK = 0U;
    if (const char *mmBaseKEnv = std::getenv("MLA_PROLOG_V3_MM_BASEK")) {
        const int baseK = std::atoi(mmBaseKEnv);
        if (baseK > 0) {
            baseParams_->mm12BaseK = static_cast<uint32_t>(baseK);
        }
    }
""", "")
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)

P3 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h")
with io.open(P3, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""
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
""", "")
s = s.replace("""    ApplyBaseKOverride(mmCqParam_, baseParams_->mm12BaseK);
""", "")
s = s.replace("""    ApplyBaseKOverride(mmCkvKrParam_, baseParams_->mm12BaseK);
""", "")
with io.open(P3, "w", encoding="utf-8") as f:
    f.write(s)
print("mm12 baseK experiment reverted")
