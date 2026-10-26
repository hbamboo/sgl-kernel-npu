#!/usr/bin/env python3
"""Experiment: let the tiling carry an mm3 baseK override (env gated).

The mm3 (W_UQ_QR) GEMM runs with baseK=64 for bf16 and stepK=4, i.e. small L0 tiles,
and the prefill profile shows AIC MTE1 (L1->L0) at ~49% while MAC is only ~20%.
Sweeping baseK (keeping the L1 footprint constant by scaling stepK) is the cheapest
way to see whether bigger L0 tiles lift cube utilisation.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 1) tiling data: extra field
P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
with io.open(P1, encoding="utf-8") as f:
    s = f.read()
OLD = """    uint32_t cvMode;
};"""
NEW = """    uint32_t cvMode;
    uint32_t mm3BaseK; // [sgl-kernel-npu port] 0 = kernel default
};"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

# 2) tiling: read env and record it
P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """    context_->tplArgs.cvMode = cvMode;"""
NEW2 = """    context_->tplArgs.cvMode = cvMode;
    context_->tplArgs.mm3BaseK = 0U;
    if (const char *mm3Env = std::getenv("MLA_PROLOG_V3_MM3_BASEK")) {
        const int baseK = std::atoi(mm3Env);
        if (baseK > 0) {
            context_->tplArgs.mm3BaseK = static_cast<uint32_t>(baseK);
        }
    }"""
# there are two places setting cvMode (empty-tensor branch + normal); patch the normal one
cnt = s.count(OLD2)
assert cnt >= 1, cnt
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)

# 3) kernel: honour it
P3 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h")
with io.open(P3, encoding="utf-8") as f:
    s = f.read()
OLD3 = """    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == ONE_BYTE_TYPE_SIZE) ? 128 : 64;"""
NEW3 = """    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == ONE_BYTE_TYPE_SIZE) ? 128 : 64;
    // [sgl-kernel-npu port] optional baseK override (keeps the L1 footprint constant)
    const uint32_t mm3BaseKOverride = 0U; // filled below from the tiling data
    (void)mm3BaseKOverride;"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
OLD4 = """    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    mmQcQrParam_.kScale = mmQcQrParam_.k / FP8_E4M3_BLOCK_SIZE;"""
NEW4 = """    mmQcQrParam_.stepK = 4;
    // [sgl-kernel-npu port] env-driven baseK experiment: keep baseK*stepK (the L1
    // footprint) constant at the default 64*4 so only the L0 tile shape changes.
    const uint32_t mm3BaseK = baseParams_->mm3BaseK;
    if (mm3BaseK == 64U || mm3BaseK == 128U || mm3BaseK == 256U) {
        mmQcQrParam_.baseK = mm3BaseK;
        mmQcQrParam_.stepK = static_cast<uint8_t>((64U * 4U) / mm3BaseK);
    }
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    mmQcQrParam_.kScale = mmQcQrParam_.k / FP8_E4M3_BLOCK_SIZE;"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P3, "w", encoding="utf-8") as f:
    f.write(s)
print("mm3 baseK experiment plumbed")
