#!/usr/bin/env python3
"""Re-add the split-K/N arch gate bypass (env gated) to inspect its tiling.

Split-K would give each core the full N with a K slice, which is the right schedule for
GLM's narrow GEMMs (N=768/576 against 32 cores = 24-32 columns per core today).  The
kernel's split-K path was gated to DAV_9201; while enabling it faulted earlier, the
first useful step is to see whether the *tiling* it produces for our shape is sane.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        return false;
    }"""
NEW = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        // [sgl-kernel-npu port] allow the split-K/N schedule on other archs behind an
        // env flag (default off) so its tiling can be inspected.
        const char *splitKnEnv = std::getenv("MLA_PROLOG_V3_SPLIT_KN");
        if (splitKnEnv == nullptr || std::atoi(splitKnEnv) == 0) {
            return false;
        }
    }"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("split-KN inspection knob added")
