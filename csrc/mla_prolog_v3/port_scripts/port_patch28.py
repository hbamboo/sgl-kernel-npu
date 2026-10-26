#!/usr/bin/env python3
"""README: host-side overhead breakdown + failed split-K/N experiment + steady-state profile."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """* **framework floor**: any NPU kernel op costs **23.4 us** per call on this stack
  (`helloworld`, tiny `qkv_rms_norm_rope_cache`).  The op itself adds ~0.7 us on top."""
NEW = """* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,
  broken down as **16.7 us pure torch dispatcher** (measured with a host-only op) +
  **~5.3 us kernel launch** (`aclrtLaunchKernelWithHostArgs` ≈ 3.9 us) + ~1 us glue.
  Our op adds ~1.5 us on top, so the host side is essentially at the framework floor."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """msprof `PipeUtilization` on the decode case (T=1, 32 cores) shows where the remaining
time goes: MTE2 (weight DMA) 36%, **AIC scalar 31%**, MTE1 15%, MAC 9.5% - i.e. the
GEMV-shaped GEMMs are limited by scalar bookkeeping and small DMA requests, not by
the cube or by raw bandwidth."""
NEW2 = """Steady-state msprof `PipeUtilization` on the decode case (T=1, 16 cores, median task
duration **18.9 us**): AIC MTE2 **47%**, AIC MTE1 **47%**, MAC 27%, AIC scalar 34%,
fixpipe 6%; AIV scalar 30%, AIV vec 0.1%.  So the GEMV-shaped GEMMs are limited by the
L1/L0 feeding pipelines plus scalar bookkeeping, with roughly 2x headroom to the DMA
roofline - closing it needs deeper Matmul-API pipeline tuning (larger L1 stages) or a
purpose-built decode path, not a tiling knob.

Two experiments that did **not** pan out (both reverted / kept opt-in):

* enabling the op's split-K/N schedule on DAV_3510 (`IsSplitKnSupported` is hard-gated
  to DAV_9201) faults the device (`507015`), so the arch gate is intentional;
* `MLA_PROLOG_V3_MAX_STEP=128` (per-step tokens 64 -> 128) is numerically identical and
  ~5% faster at T >= 256, but the 64 cap exists for UB budget reasons on D != 128, so
  it stays an opt-in knob rather than the default."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme updated")
