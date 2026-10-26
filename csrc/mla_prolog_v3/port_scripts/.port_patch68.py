"""README: record the mm1+mm2 overlap design analysis and the acceptance runner."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## Status"""
NEW = """## Remaining prefill headroom: why mm1/mm2 overlap is a kernel project

The He sweep shows the two down-projection GEMMs are ~70% of a T=256 kernel (49 of 70 us),
and they are independent: same input, different weights, different output buffers.  They
currently run **sequentially on the same block range** (`UsedBlockParams{0, mm1BlockNum}`
then `{0, mm2BlockNum}`), so only the phase boundary serialises them.

The kernel already knows how to run two sub-GEMMs on disjoint block ranges - the
group-compute path does exactly that for mm3 (`UsedBlockParams{0, QC_CORE_NUM}` and
`{QC_CORE_NUM, QC_CORE_NUM + QR_CORE_NUM}`, with offsets derived from
`blockIdx_ - QC_CORE_NUM` in `ComputeAicOffset`).  Mirroring it for mm1/mm2 is therefore
a small offset change plus a tiling choice (`mm1BlockNum + mm2BlockNum <= aicNum`, e.g.
16 + 16 so each GEMM serves a larger N per core).

What makes it a project rather than a patch is the cross-core flag protocol:
`CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ)` is phase-coupled to the
vector side, and `enableGroupComputeOpt` exists precisely to make a second, disjoint
phase legal - it is gated to int8 half quant with `G=8` (`GROUP_COMPUTE_*`) and its
template combination does not compile for our dtype on this toolchain (see the porting
notes).  A bf16 overlap therefore needs that protocol replicated (or a new flag scheme),
which is upstream-scale kernel work with deadlock risk on a shared card.

Expected payoff if done: up to ~24 us of the 70 us prefill kernel at T=256 (perfect
overlap of 49 us), realistically 10-20% of prefill.

## Acceptance runner

```bash
bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5    # device id, logs in /tmp/mla_prolog_v3_acceptance
```

Runs 11 checks in one go (CPU golden at T=8/T=256, real MXFP8 layer-2 weights, bit-exact
comparison against the vendor operator, the `_out` variant, the four SGLang shim tests,
the MXFP8 scheme-state reconstruction, and the performance sweep) and prints a summary.
Last run: 11/11 pass, decode 36.1 us (`_out`), T=2048 650 us.

## Status"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme: overlap analysis + acceptance runner")
