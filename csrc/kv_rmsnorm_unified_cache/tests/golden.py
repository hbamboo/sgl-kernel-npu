"""Independent CPU golden and framework-shaped helpers for the unified-576 cache op.

The golden follows the source operator's semantics, not the migrated implementation:
  * RMSNorm over the first 512 elements in fp32, epsilon added after dividing by 512;
  * RoPE over the last 64 elements: de-interleave even/odd, use cos/sin first/second
    32-wide halves independently, and store [even_results | odd_results];
  * BF16/FP16 rounding only on store.
"""

import math
import torch

KV_LATENT = 512
KV_ROPE = 64
KV_WIDTH = 576
PAGE = 128


def cpu_golden(kv2d, gamma, cos4d, sin4d, eps):
    """Return (latent[T,512], rope[T,64]) in the payload dtype of kv2d."""
    dtype = kv2d.dtype
    x = kv2d[:, :KV_LATENT].to(torch.float32)
    var = (x * x).sum(dim=-1) / float(KV_LATENT)
    rstd = torch.rsqrt(var + float(eps))
    latent = (x * rstd[:, None] * gamma.to(torch.float32)[None, :]).to(dtype)

    r = kv2d[:, KV_LATENT:].to(torch.float32)
    even = r[:, 0::2]
    odd = r[:, 1::2]
    cos = cos4d.reshape(cos4d.shape[0], KV_ROPE).to(torch.float32)
    sin = sin4d.reshape(sin4d.shape[0], KV_ROPE).to(torch.float32)
    c0, c1 = cos[:, : KV_ROPE // 2], cos[:, KV_ROPE // 2 :]
    s0, s1 = sin[:, : KV_ROPE // 2], sin[:, KV_ROPE // 2 :]
    out_even = even * c0 - odd * s0
    out_odd = odd * c1 + even * s1
    rope = torch.cat((out_even, out_odd), dim=-1).to(dtype)
    return latent, rope


def expected_cache(cache_base, latent, rope, index):
    """Apply golden rows to a clone at valid slots; -1/OOB skip. Last writer wins."""
    cap = cache_base.numel() // KV_WIDTH
    out = cache_base.clone()
    flat = out.view(-1, KV_WIDTH)
    for t in range(index.numel()):
        slot = int(index[t])
        if 0 <= slot < cap:
            flat[slot, :KV_LATENT] = latent[t]
            flat[slot, KV_LATENT:] = rope[t]
    return out


def callsite_cos_sin(T, rotary_dim=KV_ROPE, seed=0):
    """Shape used by the framework: cos[col] == cos(col % 32) in both halves."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    freq = torch.rand(KV_ROPE // 2, generator=g) * 6.0
    pos = torch.arange(T, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    half = torch.cos(ang)
    return torch.cat((half, half), dim=-1).reshape(T, 1, 1, KV_ROPE)


def callsite_sin(T, rotary_dim=KV_ROPE, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    freq = torch.rand(KV_ROPE // 2, generator=g) * 6.0
    pos = torch.arange(T, dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    half = torch.sin(ang)
    return torch.cat((half, half), dim=-1).reshape(T, 1, 1, KV_ROPE)


def make_indices(T, pages, mode="shuffle", seed=0, pad_dups=0):
    cap = pages * PAGE
    g = torch.Generator().manual_seed(seed + 7919)
    if mode == "shuffle":
        vals = torch.randperm(cap, generator=g)[:T]
    elif mode == "bounds":
        anchors = [0, PAGE - 1, PAGE, cap - 1]
        anchors = list(dict.fromkeys(a for a in anchors if a < cap))
        if T < len(anchors):
            vals = torch.tensor(anchors[:T], dtype=torch.long)
        else:
            pool = [v for v in torch.randperm(cap, generator=g).tolist() if v not in anchors]
            vals = torch.tensor(anchors + pool[: T - len(anchors)], dtype=torch.long)
    elif mode == "padding":
        vals = torch.zeros(T, dtype=torch.long)
        if pad_dups:
            vals[-pad_dups:] = torch.randperm(cap, generator=g)[:pad_dups]
    else:
        raise ValueError(mode)
    if T >= 4:
        vals[1] = -1
        vals[T // 2] = -1
    return vals.to(torch.int64)
