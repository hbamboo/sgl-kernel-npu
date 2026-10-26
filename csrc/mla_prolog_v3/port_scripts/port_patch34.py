#!/usr/bin/env python3
"""Fix the mm3 baseK plumbing: the field must live in MlaPrologBaseParams (what the
kernel receives), not in the tplArgs record."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(ROOT, "csrc/mla_prolog/op_kernel/mla_prolog_tiling_data.h")
P2 = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

with io.open(P1, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    uint32_t cvMode;
    uint32_t mm3BaseK; // [sgl-kernel-npu port] 0 = kernel default
};""", """    uint32_t cvMode;
};""")
OLD = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements"""
NEW = """    uint64_t kvCacheStride0; // kv cache first-axis stride in elements
    uint64_t krCacheStride0; // kr cache first-axis stride in elements
    uint32_t mm3BaseK;       // [sgl-kernel-npu port] 0 = kernel default (baseK sweep)"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P1, "w", encoding="utf-8") as f:
    f.write(s)

with io.open(P2, encoding="utf-8") as f:
    s = f.read()
s = s.replace("""    context_->tplArgs.mm3BaseK = 0U;
    if (const char *mm3Env = std::getenv("MLA_PROLOG_V3_MM3_BASEK")) {
        const int baseK = std::atoi(mm3Env);
        if (baseK > 0) {
            context_->tplArgs.mm3BaseK = static_cast<uint32_t>(baseK);
        }
    }""", "")
OLD2 = """    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
NEW2 = """    baseParams_->mm3BaseK = 0U;
    if (const char *mm3Env = std::getenv("MLA_PROLOG_V3_MM3_BASEK")) {
        const int baseK = std::atoi(mm3Env);
        if (baseK > 0) {
            baseParams_->mm3BaseK = static_cast<uint32_t>(baseK);
        }
    }
    baseParams_->kvCacheStride0 = context_->kvCacheStride0;"""
assert s.count(OLD2) == 1, s.count(OLD2)
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("mm3 baseK moved to baseParams")
