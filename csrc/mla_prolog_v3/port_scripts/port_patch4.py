#!/usr/bin/env python3
"""Port patch #4: per-tensor scale dtypes on the host tensor description.

MXFP8 mode needs E8M0 weight/dequant scales while other modes use FLOAT, so the
single `scaleType` field is not enough.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def patch(rel, pairs):
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8") as f:
        s = f.read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit("pattern count %d in %s for %r" % (n, rel, old[:70]))
        s = s.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("patched", rel)


patch("csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.h", [
    ("""    ge::DataType scaleType{ge::DT_FLOAT};""",
     """    ge::DataType scaleType{ge::DT_FLOAT};
    ge::DataType scaleXType{ge::DT_FLOAT};
    ge::DataType scaleWDqType{ge::DT_FLOAT};
    ge::DataType scaleWUqQrType{ge::DT_FLOAT};
    ge::DataType scaleWDkvKrType{ge::DT_FLOAT};
    ge::DataType scaleQNopeType{ge::DT_FLOAT};
    ge::DataType scaleQNormType{ge::DT_FLOAT};
    ge::DataType queryNormType{ge::DT_BF16};"""),
])

patch("csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp", [
    ("""    Fill(ctx.dequantScaleX.desc, ctx.dequantScaleX.shape, sScaleX, tensors.dequantScaleX, tensors.scaleType);
    Fill(ctx.dequantScaleWDq.desc, ctx.dequantScaleWDq.shape, sScaleWDq, tensors.dequantScaleWDq,
         tensors.scaleType);
    Fill(ctx.dequantScaleWUqQr.desc, ctx.dequantScaleWUqQr.shape, sScaleWUqQr, tensors.dequantScaleWUqQr,
         tensors.scaleType);
    Fill(ctx.dequantScaleWDkvKr.desc, ctx.dequantScaleWDkvKr.shape, sScaleWDkvKr, tensors.dequantScaleWDkvKr,
         tensors.scaleType);
    Fill(ctx.quantScaleCkv.desc, ctx.quantScaleCkv.shape, sQuantCkv, tensors.quantScaleCkv, tensors.scaleType);
    Fill(ctx.quantScaleCkr.desc, ctx.quantScaleCkr.shape, sQuantCkr, tensors.quantScaleCkr, tensors.scaleType);
    Fill(ctx.smoothScalesCq.desc, ctx.smoothScalesCq.shape, sSmoothCq, tensors.smoothScalesCq, tensors.scaleType);""",
     """    Fill(ctx.dequantScaleX.desc, ctx.dequantScaleX.shape, sScaleX, tensors.dequantScaleX, tensors.scaleXType);
    Fill(ctx.dequantScaleWDq.desc, ctx.dequantScaleWDq.shape, sScaleWDq, tensors.dequantScaleWDq,
         tensors.scaleWDqType);
    Fill(ctx.dequantScaleWUqQr.desc, ctx.dequantScaleWUqQr.shape, sScaleWUqQr, tensors.dequantScaleWUqQr,
         tensors.scaleWUqQrType);
    Fill(ctx.dequantScaleWDkvKr.desc, ctx.dequantScaleWDkvKr.shape, sScaleWDkvKr, tensors.dequantScaleWDkvKr,
         tensors.scaleWDkvKrType);
    Fill(ctx.quantScaleCkv.desc, ctx.quantScaleCkv.shape, sQuantCkv, tensors.quantScaleCkv, tensors.scaleType);
    Fill(ctx.quantScaleCkr.desc, ctx.quantScaleCkr.shape, sQuantCkr, tensors.quantScaleCkr, tensors.scaleType);
    Fill(ctx.smoothScalesCq.desc, ctx.smoothScalesCq.shape, sSmoothCq, tensors.smoothScalesCq, tensors.scaleType);"""),
    ("""    Fill(ctx.dequantScaleQNope.desc, ctx.dequantScaleQNope.shape, sScaleQNope, tensors.dequantScaleQNope,
         tensors.scaleType);
    Fill(ctx.queryNorm.desc, ctx.queryNorm.shape, sQueryNorm, tensors.queryNorm, tensors.queryType);
    Fill(ctx.dequantScaleQNorm.desc, ctx.dequantScaleQNorm.shape, sScaleQNorm, tensors.dequantScaleQNorm,
         tensors.scaleType);""",
     """    Fill(ctx.dequantScaleQNope.desc, ctx.dequantScaleQNope.shape, sScaleQNope, tensors.dequantScaleQNope,
         tensors.scaleQNopeType);
    Fill(ctx.queryNorm.desc, ctx.queryNorm.shape, sQueryNorm, tensors.queryNorm, tensors.queryNormType);
    Fill(ctx.dequantScaleQNorm.desc, ctx.dequantScaleQNorm.shape, sScaleQNorm, tensors.dequantScaleQNorm,
         tensors.scaleQNormType);"""),
])

patch("csrc/mla_prolog_v3/op_host/mla_prolog_v3_tiling_test.cpp", [
    ("""        if (wqMode == 3) {
            t.scaleType = ge::DT_FLOAT8_E8M0;
            t.dequantScaleX = {T, He / 32};
            t.dequantScaleWDq = {1, Hcq / 32};
            t.dequantScaleWUqQr = {1, N * (D + Dr) / 32};
            t.dequantScaleWDkvKr = {1, (Hckv + Dr) / 32};
            t.queryNorm = {T, Hcq};
            t.dequantScaleQNorm = {T, Hcq / 32};
        }""",
     """        if (wqMode == 3) {
            // MXFP8: weight scales are E8M0, laid out as [out, in/32]
            t.scaleWDqType = ge::DT_FLOAT8_E8M0;
            t.scaleWUqQrType = ge::DT_FLOAT8_E8M0;
            t.scaleWDkvKrType = ge::DT_FLOAT8_E8M0;
            t.dequantScaleX = {T, He / 32};
            t.dequantScaleWDq = {Hcq, He / 32};
            t.dequantScaleWUqQr = {N * (D + Dr), Hcq / 32};
            t.dequantScaleWDkvKr = {Hckv + Dr, He / 32};
            t.queryNormType = ge::DT_FLOAT8_E4M3FN;
            t.queryNorm = {T, Hcq};
            t.dequantScaleQNorm = {T, Hcq / 32};
            t.scaleQNormType = ge::DT_FLOAT8_E8M0;
            t.scaleQNopeType = ge::DT_FLOAT;
        }"""),
])
print("done")
