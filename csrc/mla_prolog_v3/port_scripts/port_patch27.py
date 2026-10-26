#!/usr/bin/env python3
"""Revert the split-K/N experiment: the DAV_3510 device faults (507015) with it.

Keeping the knob would be a footgun, so the upstream arch gate is restored.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

NEW = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        return false;
    }"""
OLD = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        // [sgl-kernel-npu port] experimental: allow the split-K/N schedule on other
        // archs (Ascend950) behind an env flag; default behaviour is unchanged.
        const char *splitKnEnv = std::getenv("MLA_PROLOG_V3_SPLIT_KN");
        if (splitKnEnv == nullptr || std::atoi(splitKnEnv) == 0) {
            return false;
        }
    }"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("split-KN experiment reverted (arch gate restored)")
