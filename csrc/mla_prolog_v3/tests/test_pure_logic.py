"""Device-free unit tests for the adaptation logic (no NPU tensor is ever allocated).

Covers the pure helpers that implement the GLM-4.7-Flash specifics:
  * MXFP8 -> bf16 dequantisation (both weight orientations, e8m0 handling)
  * checkpoint -> kernel orientation detection
  * rebuilding the e8m0 scale table from the MXFP8 scheme's [in/2b, out, 2] view
  * weight_quant_mode inference
  * the kv_b_proj marker SGLang looks for

    python3 csrc/mla_prolog_v3/tests/test_pure_logic.py
"""
import sys

import torch

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    _is_mxfp8,
    _scale_from_inv,
    dequant_mxfp8,
    infer_weight_quant_mode,
    reorient_to_abi,
    with_kv_b_proj_ignored,
)

BLOCK = 32
FAILS = []


def check(name, cond):
    print("%-52s %s" % (name, "OK" if cond else "FAIL"))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- dequantisation
out_f, in_f = 8, 64
w = torch.randn(out_f, in_f).to(torch.float8_e4m3fn)          # [out, in], exactly representable
sc = torch.randint(124, 131, (out_f, in_f // BLOCK), dtype=torch.uint8)  # e8m0 exponents
ref = (w.float() * torch.pow(2.0, (sc.int() - 127).float()).repeat_interleave(BLOCK, dim=1)).to(torch.bfloat16)

check("dequant_mxfp8 [out,in] matches manual dequant", torch.equal(dequant_mxfp8(w, sc), ref))

w_abi = w.t().contiguous()                                   # kernel ABI: [in, out]
check("dequant_mxfp8 [in,out] matches transpose of the above",
      torch.equal(dequant_mxfp8(w_abi, sc), ref.t().contiguous()))

zero_sc = torch.zeros(out_f, in_f // BLOCK, dtype=torch.uint8)  # exponent 0 -> zero scale
check("e8m0 exponent 0 zeroes the weight", bool((dequant_mxfp8(w, zero_sc) == 0).all()))

check("_is_mxfp8 rejects an empty scale", not _is_mxfp8(w, torch.empty(0, dtype=torch.uint8)))
check("_is_mxfp8 rejects bf16 weights", not _is_mxfp8(ref, sc))
check("_is_mxfp8 accepts fp8 + e8m0", _is_mxfp8(w, sc))

# ---------------------------------------------------------------- orientation
# fp8 tensors cannot be compared with torch.equal on CPU; compare their bytes
eq8 = lambda a, b: torch.equal(a.view(torch.uint8), b.view(torch.uint8))  # noqa: E731
check("reorient_to_abi transposes [out,in]", eq8(reorient_to_abi(w, sc), w.t().contiguous()))
check("reorient_to_abi keeps [in,out]", eq8(reorient_to_abi(w_abi, sc), w_abi))
t3 = torch.empty(4, 8, 16)
check("reorient_to_abi ignores 3-D weights", reorient_to_abi(t3, sc) is t3)
check("reorient_to_abi ignores a missing scale", eq8(reorient_to_abi(w, None), w))

# ---------------------------------------------------------------- scale reconstruction
scale_inv = sc.reshape(out_f, in_f // BLOCK // 2, 2).transpose(0, 1).contiguous()  # [in/2b, out, 2]
check("_scale_from_inv rebuilds the [out,in/32] table",
      torch.equal(_scale_from_inv(scale_inv, out_f, in_f), sc))
check("_scale_from_inv rejects a mismatched in-dim", _scale_from_inv(scale_inv, out_f, in_f * 2) is None)

# ---------------------------------------------------------------- mode inference
check("infer: bf16 -> 0", infer_weight_quant_mode(ref, None) == 0)
check("infer: fp8 + scale -> 3", infer_weight_quant_mode(w, sc) == 3)
check("infer: fp8 without scale -> None", infer_weight_quant_mode(w, None) is None)
check("infer: int8 -> None (caller decides)",
      infer_weight_quant_mode(torch.empty(4, 4, dtype=torch.int8), None) is None)

# ---------------------------------------------------------------- SGLang marker
check("marker added to an empty list", with_kv_b_proj_ignored([]) == ["kv_b_proj"])
check("marker not duplicated", with_kv_b_proj_ignored(["kv_b_proj"]) == ["kv_b_proj"])
check("marker kept after other entries",
      with_kv_b_proj_ignored(["lm_head"]) == ["lm_head", "kv_b_proj"])

print()
if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
print("all pure-logic checks passed (no NPU allocation involved)")
