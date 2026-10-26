#!/usr/bin/env python3
"""Map torch float8 dtypes to GE dtypes and allocate the empty dequant-scale
output for the non-quantised configuration."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

pairs = [
    ("""        case at::kByte: return ge::DT_UINT8;
        default: return ge::DT_BF16;""",
     """        case at::kByte: return ge::DT_UINT8;
        case at::kFloat8_e4m3fn: return ge::DT_FLOAT8_E4M3FN;
        case at::kFloat8_e5m2: return ge::DT_FLOAT8_E5M2;
        default: return ge::DT_BF16;"""),
    ("""    if (!query_norm_flag) {
        dequant_q_norm = EmptyLikeShape({0}, at::kFloat, token_x);
    } else if (weight_quant_mode == 3) {""",
     """    if (!query_norm_flag || weight_quant_mode == 0) {
        dequant_q_norm = EmptyLikeShape({0}, at::kFloat, token_x);
    } else if (weight_quant_mode == 3) {"""),
]
for old, new in pairs:
    n = s.count(old)
    if n != 1:
        raise SystemExit("pattern count %d for %r" % (n, old[:60]))
    s = s.replace(old, new)

with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("patched dtype map")
