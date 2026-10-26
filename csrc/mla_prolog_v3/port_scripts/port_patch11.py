#!/usr/bin/env python3
"""MXFP8 dtype mapping: E8M0 scales are carried as uint8 tensors (torch has no
e8m0 dtype), so map them to DT_FLOAT8_E8M0 when weight_quant_mode == 3."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

OLD = """    t.scaleXType = optDtype(dequant_scale_x);
    t.scaleWDqType = optDtype(dequant_scale_w_dq);
    t.scaleWUqQrType = optDtype(dequant_scale_w_uq_qr);
    t.scaleWDkvKrType = optDtype(dequant_scale_w_dkv_kr);"""

NEW = """    t.scaleXType = optDtype(dequant_scale_x);
    t.scaleWDqType = optDtype(dequant_scale_w_dq);
    t.scaleWUqQrType = optDtype(dequant_scale_w_uq_qr);
    t.scaleWDkvKrType = optDtype(dequant_scale_w_dkv_kr);
    if (weight_quant_mode == 3) {
        // MXFP8: scale tensors are FLOAT8_E8M0, carried as uint8 on the torch side.
        t.scaleXType = ge::DT_FLOAT8_E8M0;
        t.scaleWDqType = ge::DT_FLOAT8_E8M0;
        t.scaleWUqQrType = ge::DT_FLOAT8_E8M0;
        t.scaleWDkvKrType = ge::DT_FLOAT8_E8M0;
        t.scaleQNormType = ge::DT_FLOAT8_E8M0;
    }"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s.replace(OLD, NEW))
print("patched scale dtypes")
