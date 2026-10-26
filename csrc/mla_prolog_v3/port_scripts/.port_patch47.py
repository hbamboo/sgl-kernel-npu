#!/usr/bin/env python3
"""Let the real-weight test run through the SGLang-facing alias (SGLANG_ALIAS=1)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/tests/test_real_glm_weights.py")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """nz = lambda t: torch_npu.npu_format_cast(t.clone().to(dev), 29)
q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(
    x, nz(q_a), nz(q_b), w_kc.to(dev), nz(kv_a), gamma_cq.to(dev), gamma_ckv.to(dev), sin, cos, kvc, krc,
    cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
    weight_quant_mode=0, kv_cache_quant_mode=0)
torch_npu.npu.synchronize()"""

NEW = """nz = lambda t: torch_npu.npu_format_cast(t.clone().to(dev), 29)

# exactly the kwargs SGLang's forward_mlaprolog builds
mlaprolog_args = dict(
    token_x=x,
    weight_dq=nz(q_a),
    weight_uq_qr=nz(q_b),
    weight_uk=w_kc.to(dev),
    weight_dkv_kr=nz(kv_a),
    rmsnorm_gamma_cq=gamma_cq.to(dev),
    rmsnorm_gamma_ckv=gamma_ckv.to(dev),
    rope_sin=sin,
    rope_cos=cos,
    kv_cache=kvc,
    kr_cache=krc,
    cache_index=idx,
    cache_mode="PA_BSND",
    query_norm_flag=True,
    weight_quant_mode=0,
)

if os.environ.get("SGLANG_ALIAS"):
    import sys

    sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
    from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias

    print("registered custom::npu_mla_prolog_v3:", register_custom_alias())
    q, qr, dqn, qn, dqnorm = torch.ops.custom.npu_mla_prolog_v3(**mlaprolog_args)
else:
    q, qr, dqn, qn, dqnorm = torch.ops.npu.mla_prolog_v3(**mlaprolog_args)
torch_npu.npu.synchronize()"""

assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("real-weight test can run through the SGLang alias")
