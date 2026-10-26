"""Verify the SGLang-facing alias resolves and matches the direct kernel call.

Reproduces the exact kwargs SGLang's ``forward_mlaprolog`` builds (mla_preprocess.py)
against ``torch.ops.custom.npu_mla_prolog_v3`` after loading the compat shim, and
compares with a direct ``torch.ops.npu.mla_prolog_v3`` call.
"""
import sys

import torch
import torch_npu

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")
import sgl_kernel_npu  # noqa: F401,E402  (loads libsgl_kernel_npu.so)
from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias  # noqa: E402

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 2048, 768, 512, 192, 64, 20
T = 8
torch.manual_seed(0)

print("alias registered:", register_custom_alias())

gq = torch.rand(Hcq, dtype=torch.bfloat16, device=dev) + 0.5
gkv = torch.rand(Hckv, dtype=torch.bfloat16, device=dev) + 0.5
Wdq = torch_npu.npu_format_cast(torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuqqr = torch_npu.npu_format_cast(torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wdkv = torch_npu.npu_format_cast(torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev) * 0.02, 29)
Wuk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev) * 0.02
x = torch.randn(T, He, dtype=torch.bfloat16, device=dev) * 0.1

pos = torch.arange(T)
inv = 1.0 / (1000000.0 ** (torch.arange(0, Dr, 2).float() / Dr))
freq = torch.outer(pos.float(), inv)
cos = torch.cat([freq.cos(), freq.cos()], 1).to(torch.bfloat16).to(dev)
sin = torch.cat([freq.sin(), freq.sin()], 1).to(torch.bfloat16).to(dev)
blocks = (T + 127) // 128 + 1
kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
idx = torch.arange(T, dtype=torch.int64, device=dev)

# exactly what SGLang forward_mlaprolog passes
mla_prolog_input_args = {
    "token_x": x,
    "weight_dq": Wdq,
    "weight_uq_qr": Wuqqr,
    "weight_uk": Wuk,
    "weight_dkv_kr": Wdkv,
    "rmsnorm_gamma_cq": gq,
    "rmsnorm_gamma_ckv": gkv,
    "rope_sin": sin,
    "rope_cos": cos,
    "kv_cache": kvc,
    "kr_cache": krc,
    "cache_index": idx,
    "rmsnorm_epsilon_cq": 1e-5,
    "rmsnorm_epsilon_ckv": 1e-5,
    "cache_mode": "PA_BSND",
    "query_norm_flag": True,
    "weight_quant_mode": 0,
}

q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = torch.ops.custom.npu_mla_prolog_v3(
    **mla_prolog_input_args
)
torch_npu.npu.synchronize()

ref = torch.ops.npu.mla_prolog_v3(**mla_prolog_input_args)
torch_npu.npu.synchronize()

pairs = [("q_nope", q_nope, ref[0]), ("q_pe", q_pe, ref[1]), ("qr", qr, ref[3])]
for name, got, want in pairs:
    same = torch.equal(got, want)
    cs = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), 0).item() if got.numel() else 1.0
    print("%-8s shape=%-18s bit-identical=%-5s cosine=%.7f" % (name, tuple(got.shape), same, cs))
print("dequant_scale_q_nope:", tuple(dequant_scale_q_nope.shape),
      " dequant_q_norm:", tuple(dequant_q_norm.shape))
print("kv_cache written tokens:", int((kvc.abs().sum(-1).sum(-1).sum(-1) > 0).sum()),
      " kr_cache:", int((krc.abs().sum(-1).sum(-1).sum(-1) > 0).sum()))
