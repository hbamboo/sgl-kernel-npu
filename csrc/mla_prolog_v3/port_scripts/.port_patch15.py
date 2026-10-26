#!/usr/bin/env python3
"""Add an opt-in core-count override for tuning: MLA_PROLOG_V3_CORES=<n>."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """    OP_CHECK_IF((aicNum_ != aivNum_) && (aicNum_ * 2 != aivNum_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "aicNum(%u):aivNum(%u) only support 1:1 or 1:2", aicNum_,
                                            aivNum_),
                return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;"""

NEW = """    OP_CHECK_IF((aicNum_ != aivNum_) && (aicNum_ * 2 != aivNum_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "aicNum(%u):aivNum(%u) only support 1:1 or 1:2", aicNum_,
                                            aivNum_),
                return GRAPH_FAILED);

    // [sgl-kernel-npu port] optional core-count override for tuning: the tiling was
    // sized for the DeepSeek geometry, GLM shapes are ~3x smaller.
    const char *coresEnv = std::getenv("MLA_PROLOG_V3_CORES");
    if (coresEnv != nullptr) {
        const int cores = std::atoi(coresEnv);
        if (cores > 0) {
            aicNum_ = static_cast<uint32_t>(cores);
            aivNum_ = aicNum_ * 2U;
        }
    }

    return ge::GRAPH_SUCCESS;"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)
if "#include <cstdlib>" not in s:
    s = s.replace("#include <numeric>", "#include <cstdlib>\n#include <numeric>")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("core override added")
