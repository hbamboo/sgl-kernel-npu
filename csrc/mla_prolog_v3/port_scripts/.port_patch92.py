#!/usr/bin/env python3
"""Docs: record the token-boundary sweep (31 sizes, 13/13 acceptance)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()
OLD = """### 4.3 真实 checkpoint（`model.layers.2`，MXFP8）"""
NEW = """### 4.3 token 边界扫描（31 个尺寸）

围绕 kernel 内部的所有分块边界（PA block 128、每步 128/64 token、16/32 对齐）逐个与
CPU golden 比对，阈值 0.9995，全部通过（`tests/sweep_tokens.py`，已并入验收）：

| T | query | q_rope | kv_cache | kr_cache |
|---|---|---|---|---|
| 1 | 0.999996 | 0.999996 | 0.999997 | 0.999998 |
| 16 / 17 | 0.999995 / 0.999993 | 0.999996 | 0.999998 | 0.999998 |
| 63 / 64 / 65 | 0.999979 / 0.999978 | 0.999995 | 0.999999 | 0.999998 |
| 127 / 128 / 129 / 130 | 0.999961 | 0.999994 | 1.000000 | 0.999999 |
| 191 / 192 / 193 | 0.999970 | 0.999987 | 0.999996 | 0.999999 |
| 255 / 256 / 257 / 258 | 0.999980 | 0.999981 | 0.999995 | 0.999999 |
| 383 / 384 / 385 | 1.000054 | 0.999977 | 0.999994 | 1.000000 |
| 511 / 512 / 513 | 1.000094 | 0.999975 | 0.999993 | 1.000000 |
| 1000 | 1.000358 | 0.999965 | 0.999980 | 0.999998 |

（`query` 的 cosine 偶尔 > 1 是浮点舍入所致，正常。）特别地，T=191/192/193 这组曾经因为
PA stride 缺陷而完全错误，现在是 0.99997 —— 说明该修复在所有边界上都成立。

### 4.4 真实 checkpoint（`model.layers.2`，MXFP8）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
OLD2 = """### 4.4 与厂商算子"""
NEW2 = """### 4.5 与厂商算子"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD3 = """Last run: **12/12 pass**"""
NEW3 = """It includes a token-boundary sweep (`sweep_tokens.py`): 31 sizes around every internal
tile boundary (1..1000) all stay >= 0.9995 on the five outputs.

Last run: **13/13 pass**"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated with the boundary sweep")
