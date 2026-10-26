"""Revert the split-K/N gate bypass and correct the overlap analysis in the README.

Findings this round:
  * with the gate bypassed the tiling still picks the plain N split (the split-K/N search
    finds no candidate for GLM shapes), so the knob is a no-op - reverted;
  * the "overlap mm1+mm2" idea has no headroom: both phases already use 24 of 32 cores
    and are throughput-bound, so moving mm2 onto the 8 idle cores would need
    ceil(576/8) = 72 columns/core, i.e. ~3x the per-core work, making it slower than the
    sequential 18 us.  Overlap only helps when a phase leaves cores idle *and* the other
    phase can use them without losing efficiency.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling.cpp")
with io.open(P, encoding="utf-8") as f:
    s = f.read()
NEW = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
        return false;
    }"""
OLD = """    if (GetCurNpuArch() != NpuArch::DAV_9201) {
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

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """Expected payoff if done: up to ~24 us of the 70 us prefill kernel at T=256 (perfect
overlap of 49 us), realistically 10-20% of prefill."""
NEW2 = """Measured follow-up (this was checked, not assumed): the overlap has **no headroom**.
At T=256 mm1 uses 24 of 32 cores (32 columns each) and mm2 uses 24 (24 columns each);
both are throughput-bound, so moving mm2 onto the 8 idle cores would need 576/8 = 72
columns per core - roughly 3x the per-core work - and end up slower than the 18 us it
takes sequentially.  The only real lever is split-K (full N per core, K sliced), whose
search (`mla_prolog_kn_search.h`) returns no candidate for GLM's shapes and whose kernel
path is gated to DAV_9201.  Bypassing that gate changes nothing in the tiling, which
confirms the schedule simply is not available for this geometry."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("knob reverted; overlap analysis corrected")
