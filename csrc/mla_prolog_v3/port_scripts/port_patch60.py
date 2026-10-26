"""Shim: dump the op arguments when tiling fails (debug aid for the e2e run)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = '''    return torch.ops.npu.mla_prolog_v3(*args, **kwargs)'''
NEW = '''    try:
        return torch.ops.npu.mla_prolog_v3(*args, **kwargs)
    except Exception:
        if os.environ.get("DSH_MLA_PROLOG_V3_DEBUG") == "1":
            names = ["token_x", "weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr",
                     "rmsnorm_gamma_cq", "rmsnorm_gamma_ckv", "rope_sin", "rope_cos",
                     "kv_cache", "kr_cache"]
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    try:
                        fmt = torch_npu.get_npu_format(a)
                    except Exception:  # noqa: BLE001
                        fmt = "?"
                    print("[dsh-shim] %-18s %-22s %-16s fmt=%s"
                          % (names[i] if i < len(names) else "arg%d" % i,
                             tuple(a.shape), str(a.dtype).replace("torch.", ""), fmt), flush=True)
            for k, v in sorted(kwargs.items()):
                if isinstance(v, torch.Tensor):
                    print("[dsh-shim] %-26s %-22s %s" % (k, tuple(v.shape),
                                                         str(v.dtype).replace("torch.", "")), flush=True)
                else:
                    print("[dsh-shim] %-26s %s" % (k, v), flush=True)
        raise'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

if "\nimport os\n" not in s:
    s = s.replace("import re\n\nimport torch", "import os\nimport re\n\nimport torch")
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("debug dump added")
