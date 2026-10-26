"""Add an env-gated tiling-only probe so the host cost can be split precisely."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """    // the launch macro captures its blockdim argument by name in a lambda, so it must
    // be a plain local variable
    const uint32_t block_dim = res.blockDim;"""
NEW = """    // profiling aid: MLA_PROLOG_V3_PROBE_TILING=1 stops right before the kernel launch
    if (std::getenv("MLA_PROLOG_V3_PROBE_TILING") != nullptr) {
        return;
    }

    // the launch macro captures its blockdim argument by name in a lambda, so it must
    // be a plain local variable
    const uint32_t block_dim = res.blockDim;"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

if "#include <cstdlib>" not in s:
    s = s.replace("#include <mutex>", "#include <cstdlib>\n#include <mutex>")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("tiling probe added")
