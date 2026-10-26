"""Check the mlaprolog routing helper and that sglang is patchable in this image."""
import re
import sys

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
from sgl_kernel_npu.mla_prolog_v3_compat import (  # noqa: E402
    register_custom_alias,
    with_kv_b_proj_ignored,
    enable_mlaprolog,
)

# 1) the marker logic must satisfy SGLang's predicate
cases = [
    ([], True),
    (["lm_head"], True),
    (["kv_b_proj"], False),
    (["model.layers.3.self_attn.kv_b_proj"], False),
]
ok = True
for ignore, should_add in cases:
    out = with_kv_b_proj_ignored(ignore)
    matches = any(re.fullmatch(r".*kv_b_proj", x) for x in out)
    added = len(out) == len(ignore) + (1 if should_add else 0)
    ok &= matches and added
    print("ignore=%-45s -> %-50s predicate=%s" % (ignore, out, matches))
print("marker logic:", "OK" if ok else "FAILED")

# 2) alias registration
print("custom alias registered:", register_custom_alias())

# 3) sglang is importable here? the routing patch needs it
patched = enable_mlaprolog()
print("enable_mlaprolog() ->", patched)
if patched:
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig

    print("ModelSlimConfig patched:", getattr(ModelSlimConfig, "_mlaprolog_patched", False))
