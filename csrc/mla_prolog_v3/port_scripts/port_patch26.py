#!/usr/bin/env python3
"""Experimental: allow the split-K/N tiling path on DAV_3510 (env gated).

`IsSplitKnSupported()` is hard-gated to DAV_9201, so on Ascend950 the mm1/mm2 GEMMs
fall back to a pure N split over 12/9 cores (N per core is aligned to 64), leaving
most of the cube array idle.  The split-K/N schedule is a runtime tiling decision
(the kernel branches on mm1NGroup/mm1KGroup), so it can be tried on 3510 with an env
flag, keeping the default behaviour untouched.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

OLD = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        return false;
    }"""
NEW = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
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
print("split-KN experiment knob added")
