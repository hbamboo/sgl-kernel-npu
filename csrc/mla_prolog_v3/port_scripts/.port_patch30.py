#!/usr/bin/env python3
"""README: document the mm1 min-tile knob and the measurement-noise caveat."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """* `MLA_PROLOG_V3_MAX_STEP=128` (per-step tokens 64 -> 128) is numerically identical and
  ~5% faster at T >= 256, but the 64 cap exists for UB budget reasons on D != 128, so
  it stays an opt-in knob rather than the default."""
NEW = """* `MLA_PROLOG_V3_MAX_STEP=128` (per-step tokens 64 -> 128) is numerically identical and
  ~5% faster at T >= 256, but the 64 cap exists for UB budget reasons on D != 128, so
  it stays an opt-in knob rather than the default.
* `MLA_PROLOG_V3_MM1_MIN_TILE` (default 64) changes how many cores the mm1 GEMM uses
  (64 -> 12 cores for Hcq=768).  Setting it to 48 measured a reproducible but small
  ~1.2% gain at T >= 256 (94.7 vs 96.0 us, 682 vs 689 us); at decode the run-to-run
  noise (+-1.5 us, occasionally +-8 us on a shared card) hides any effect, so the
  upstream default is kept."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme updated")
