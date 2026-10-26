#!/usr/bin/env python3
"""Revert the mm3 baseK experiment: it measured no effect (93.8 vs 93.9 vs 93.9 us),
so mm3's L0 tile shape is not the prefill bottleneck and the dead knob is removed."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""
    uint32_t mm3BaseK;       // [sgl-kernel-npu port] 0 = kernel default (baseK sweep)""", "")
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    baseParams_->mm3BaseK = 0U;
    if (const char *mm3Env = std::getenv("MLA_PROLOG_V3_MM3_BASEK")) {
        const int baseK = std::atoi(mm3Env);
        if (baseK > 0) {
            baseParams_->mm3BaseK = static_cast<uint32_t>(baseK);
        }
    }
""", "")
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)

P3 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h")
with io.open(P3, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    // [sgl-kernel-npu port] optional baseK override (keeps the L1 footprint constant)
    const uint32_t mm3BaseKOverride = 0U; // filled below from the tiling data
    (void)mm3BaseKOverride;
""", "")
s = s.replace("""    mmQcQrParam_.stepK = 4;
    // [sgl-kernel-npu port] env-driven baseK experiment: keep baseK*stepK (the L1
    // footprint) constant at the default 64*4 so only the L0 tile shape changes.
    const uint32_t mm3BaseK = baseParams_->mm3BaseK;
    if (mm3BaseK == 64U || mm3BaseK == 128U || mm3BaseK == 256U) {
        mmQcQrParam_.baseK = mm3BaseK;
        mmQcQrParam_.stepK = static_cast<uint8_t>((64U * 4U) / mm3BaseK);
    }
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;""",
              """    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;""")
with io.open(P3, "w", encoding="utf-8") as f:
    f.write(s)
print("mm3 baseK experiment reverted")
