#!/usr/bin/env python3
"""Fix kernel dispatch macro: argument order must match the kernel signature and
`tiling` must be forwarded."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp")

OLD = """#define MLA_PROLOG_V3_ARGS                                                                                          \\
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,          \\
        cacheIndex, kvCache, krCache, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,         \\
        quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut,          \\
        kvCacheOut, krCacheOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace"""

NEW = """#define MLA_PROLOG_V3_ARGS                                                                                          \\
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, \\
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv,   \\
        quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCacheOut, krCacheOut, \\
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1, s.count(OLD)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s.replace(OLD, NEW))
print("fixed kernel arg macro")
