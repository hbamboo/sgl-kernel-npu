"""Partial golden check for the bf16 path: c_Q = alpha * RMSNorm(x @ W_DQ).

Compares the kernel's query_norm output (which is exactly c_Q in the
non-quantised configuration) against a CPU reference.
"""
import torch

d = torch.load("/tmp/mla_prolog_smoke.pt")
x = d["x"].float()
Wdq = d["Wdq"].float()
gq = d["gq"].float()
q_norm = d["qn"].float()
eps = 1e-5

cq = x @ Wdq
rms = torch.sqrt(cq.pow(2).mean(dim=-1, keepdim=True) + eps)
cq = cq / rms * gq

diff = (q_norm - cq).abs()
denom = cq.abs().clamp_min(1e-6)
print("c_Q reference mean|.| =", cq.abs().mean().item())
print("kernel      mean|.| =", q_norm.abs().mean().item())
print("max abs diff =", diff.max().item())
print("mean rel diff =", (diff / denom).mean().item())
print("cosine =", torch.nn.functional.cosine_similarity(q_norm.flatten(), cq.flatten(), dim=0).item())

# also check the kv cache (c_KV) against the reference
Wdkv = d["Wdkv"].float()
gkv = d["gkv"].float()
Hckv = 512
ckv = x @ Wdkv[:, :Hckv]
rms = torch.sqrt(ckv.pow(2).mean(dim=-1, keepdim=True) + eps)
ckv = ckv / rms * gkv
kvc = d["kvc"].float()
idx = d["idx"].long()
got = kvc.view(-1, Hckv)[idx]
print("kv max abs diff =", (got - ckv).abs().max().item())
print("kv cosine =", torch.nn.functional.cosine_similarity(got.flatten(), ckv.flatten(), dim=0).item())
