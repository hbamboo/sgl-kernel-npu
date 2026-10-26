#!/usr/bin/env python3
"""Docs: multi-layer real-weight validation (layers 0/1/2/23/46) and 16-check acceptance."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()
OLD = """### 4.4 真实 checkpoint（`model.layers.2`，MXFP8）

五路输出 cosine ≥ **0.9999975**；"""
NEW = """### 4.4 真实 checkpoint（多层，MXFP8）

不止测了一层（避免"恰好正确"）：

| layer | query | q_rope | query_norm | kv_cache | kr_cache |
|---|---|---|---|---|---|
| 0 | 0.9999969 | 0.9999982 | 0.9999980 | 0.9999978 | 0.9999983 |
| 1 | 0.9999972 | 0.9999980 | 0.9999978 | 0.9999978 | 0.9999979 |
| 2 | 0.9999975 | 0.9999979 | 0.9999979 | 0.9999980 | 0.9999982 |
| 23 | 0.9999949 | 0.9999969 | 0.9999982 | 0.9999977 | 0.9999982 |
| 46 | 0.9999971 | 0.9999980 | 0.9999983 | 0.9999980 | 0.9999982 |

五路输出 cosine ≥ **0.9999949**（首层、中间层、末层均覆盖，验收脚本自动跑 0/2/23/46）；
"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """Last run: **13/13 pass**"""
NEW2 = """It runs four real-checkpoint layers (0/2/23/46) so the validation is not tied to one layer.

Last run: **16/16 pass**"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
OLD3 = """| T | query | query_rope | kv_cache | kr_cache |"""
NEW3 = """Real checkpoint, per layer (five outputs, cosine):

| layer | 0 | 1 | 2 | 23 | 46 |
|---|---|---|---|---|---|
| worst output | 0.9999969 | 0.9999972 | 0.9999975 | 0.9999949 | 0.9999971 |

| T | query | query_rope | kv_cache | kr_cache |"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
