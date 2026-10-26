"""Cross-check the ported kernel against the CANN built-in MlaPrologV3 numerically
on the DeepSeek-like geometry that the built-in binary supports."""
import torch
import torch_npu

LIB = "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
torch.ops.load_library(LIB)

dev = "npu:0"
He, Hcq, Hckv, D, Dr, N = 7168, 1536, 512, 128, 64, 16
T = 8
torch.manual_seed(0)

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
idx = torch.arange(T, dtype=torch.int64, device=dev)


def run(fn):
    kvc = torch.zeros(blocks, 128, 1, Hckv, dtype=torch.bfloat16, device=dev)
    krc = torch.zeros(blocks, 128, 1, Dr, dtype=torch.bfloat16, device=dev)
    out = fn(kvc, krc)
    torch_npu.npu.synchronize()
    return out, kvc, krc


def ported(kvc, krc):
    return torch.ops.npu.mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True,
        weight_quant_mode=0, kv_cache_quant_mode=0)


def builtin(kvc, krc):
    return torch_npu.npu_mla_prolog_v3(
        x, Wdq, Wuqqr, Wuk, Wdkv, gq, gkv, sin, cos, kvc, krc,
        cache_index=idx, cache_mode="PA_BSND", query_norm_flag=True, weight_quant_mode=0)


p, pkv, pkr = run(ported)
b, bkv, bkr = run(builtin)
print("ported outputs:", len(p), "builtin outputs:", len(b))


def cmp(name, a, c):
    a = a.float()
    c = c.float()
    cs = torch.nn.functional.cosine_similarity(a.flatten(), c.flatten(), 0).item()
    print("%-12s cosine=%.7f  max|d|=%.5f  |ref|mean=%.4f"
          % (name, cs, (a - c).abs().max().item(), c.abs().mean().item()))


cmp("query", p[0], b[0])
cmp("query_rope", p[1], b[1])
cmp("query_norm", p[3], b[3])
cmp("kv_cache", pkv, bkv)
cmp("kr_cache", pkr, bkr)
