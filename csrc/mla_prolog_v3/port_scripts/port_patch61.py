"""Debug: dump the MLAPO module's weight state when the bf16 preparation runs."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''    fused_w = getattr(fused, "weight", None) if fused is not None else None
    if fused_w is not None and fused_w.dtype == torch.float8_e4m3fn:'''
NEW = '''    fused_w = getattr(fused, "weight", None) if fused is not None else None
    if os.environ.get("DSH_MLA_PROLOG_V3_DEBUG") == "1":
        def _info(mod, label):
            w = getattr(mod, "weight", None) if mod is not None else None
            si = getattr(mod, "weight_scale_inv", None) if mod is not None else None
            print("[dsh-prep] %-8s type=%-28s weight=%s %s  scale_inv=%s" % (
                label, type(mod).__name__ if mod is not None else "None",
                tuple(w.shape) if w is not None else None,
                str(w.dtype).replace("torch.", "") if w is not None else "-",
                tuple(si.shape) if si is not None else None), flush=True)
        _info(fused, "fused")
        _info(q_b, "q_b")
        print("[dsh-prep] q_a_proj_weight=%s kv_a_proj_weight=%s" % (
            tuple(module.q_a_proj_weight.shape) if getattr(module, "q_a_proj_weight", None) is not None else None,
            tuple(module.kv_a_proj_weight.shape) if getattr(module, "kv_a_proj_weight", None) is not None else None),
            flush=True)
    if fused_w is not None and fused_w.dtype == torch.float8_e4m3fn:'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

# also handle an already-bf16 [in, out] fused weight (no transpose needed)
OLD2 = '''    # q_b: [in, out] already transposed by the MXFP8 scheme'''
NEW2 = '''    elif fused_w is not None and fused_w.dtype == torch.bfloat16 and fused_w.shape[0] == He_guard(fused_w):
        pass  # placeholder, handled below

    # q_b: [in, out] already transposed by the MXFP8 scheme'''
# (kept simple: the bf16 fused case is handled by the branch added below)
if OLD2 in s:
    pass

OLD3 = '''    # q_b: [in, out] already transposed by the MXFP8 scheme
    if q_b is not None and getattr(q_b, "weight", None) is not None:'''
NEW3 = '''    # fused projection already in bf16 [in, out] (merged by the loader): just split it
    if fused_w is not None and fused_w.dtype == torch.bfloat16 and fused_w.dim() == 2:
        nd = fused_w
        try:
            if torch_npu.get_npu_format(nd) != 2:
                nd = torch_npu.npu_format_cast(nd, 2)
        except Exception:  # noqa: BLE001
            nd = nd
        hcq = module.q_lora_rank
        if nd.shape[1] == getattr(module, "num_local_heads", 0) * 0 + nd.shape[1] and nd.shape[1] > hcq:
            module.q_a_proj_weight = torch_npu.npu_format_cast(nd[:, :hcq].contiguous(), _NZ_FORMAT)
            module.kv_a_proj_weight = torch_npu.npu_format_cast(nd[:, hcq:].contiguous(), _NZ_FORMAT)

    # q_b: [in, out] already transposed by the MXFP8 scheme
    if q_b is not None and getattr(q_b, "weight", None) is not None:'''
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("debug + bf16-fused handling added")
