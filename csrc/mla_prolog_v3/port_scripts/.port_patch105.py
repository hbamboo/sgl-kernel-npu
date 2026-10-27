#!/usr/bin/env python3
"""Record the quiet-card re-validation of the decode core heuristic."""
import io

ROOT = "/mnt/paas/hxdnpu/sgl-kernel-npu"
P = ROOT + "/csrc/mla_prolog_v3/PERF-COUNTEREVIDENCE.md"
s = io.open(P, encoding="utf-8").read()

OLD = """| 1 | 用的核太少/太多 | `MLA_PROLOG_V3_CORES` 扫 8/16/24/32（解码 T=1，各 2 次） | 42.1/42.1(**16 最优**)、44.6/44.2、42.2/49.4、44.7/44.2 | 已固化"T≤128 用 16 核" |"""
NEW = """| 1 | 用的核太少/太多 | `MLA_PROLOG_V3_CORES` 扫 8/12/16/20/24/28/32（解码 T=1；早期 2 次、空闲复测各 4 次） | 空闲复测中位数：8→44.4、12→41.4、**16→41.4**、20→41.5、24→42.0、28→42.8、32→44.2 | 已固化"T≤128 用 16 核"，空闲卡复测确认仍最优（20 核与之相当，24 及以上变差）|"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("core-heuristic re-validation recorded")
