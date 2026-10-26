#!/usr/bin/env python3
"""Port patch #3: accept Hcq=768 (GLM-4.7-Flash q_lora_rank).

The upstream check only allows {1536, 2048}; GLM-4.7-Flash uses q_lora_rank=768.
Nothing in the tiling/kernel arithmetic requires 1536 besides this gate, so the
value is added to the supported set (no other change).
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling_check.cpp")

OLD = """        const std::set<uint32_t> supportedHcqSize{1536U, 2048U};"""
NEW = """        // [sgl-kernel-npu port] 768 is the GLM-4.7-Flash q_lora_rank.
        const std::set<uint32_t> supportedHcqSize{768U, 1536U, 2048U};"""

with io.open(PATH, encoding="utf-8") as f:
    src = f.read()
assert src.count(OLD) == 1, src.count(OLD)
src = src.replace(OLD, NEW)
with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("patched", PATH)
