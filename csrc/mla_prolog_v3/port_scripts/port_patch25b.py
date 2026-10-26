#!/usr/bin/env python3
"""Document the PA cache stride fix in README-port.md (matching the current text)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """| 13 | tiling cache (shapes/dtypes/attrs) + reusable workspace: host cost 49 → 27 us/call |"""
NEW = """| 13 | tiling cache (shapes/dtypes/attrs) + reusable workspace: host cost 49 → 24 us/call |
| 15/18/23 | tuning knobs: `MLA_PROLOG_V3_CORES` (core count) and `MLA_PROLOG_V3_MAX_STEP` (per-step token cap) |
| 24 | **fix**: supply `kvCacheStride0` / `krCacheStride0`.  `ConvertContext` derives them from the GE tensor descriptor; the direct-launch host left both at 0, so the PA scatter ignored the block stride and every token spilling into block >= 1 (T > blockSize) landed in the wrong slot |"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """Real checkpoint (`model.layers.2`, MXFP8 weights dequantised to bf16): all five
outputs cosine >= 0.9999975, max abs diff <= 0.057."""
NEW2 = """Real checkpoint (`model.layers.2`, MXFP8 weights dequantised to bf16): all five
outputs cosine >= 0.9999975, max abs diff <= 0.057.

Verified across the token range after the PA-stride fix (CPU golden, GLM geometry):

| T | query | query_rope | kv_cache | kr_cache |
|---|---|---|---|---|
| 8 | 0.999994 | 0.999996 | 0.999998 | 0.999998 |
| 128 | 0.999961 | 0.999994 | 1.000000 | 0.999999 |
| 256 | 0.999980 | 0.999982 | 0.999995 | 0.999999 |
| 512 | 1.000094 | 0.999975 | 0.999993 | 1.000000 |

Before the fix, T >= 192 produced garbage in `kv_cache` / `kr_cache` (all tokens
beyond the first PA block were written with stride 0)."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme updated")
