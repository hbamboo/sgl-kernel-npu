"""Performance comparison: AscendC qkv_rms_norm_rope_cache vs Triton split_qkv_rmsnorm_rope.

Config (user requested): 32k tokens, head_dim 128, q 32 heads, kv 8+8 heads.

Both kernels read the same packed qkv and write the same number of bytes:
  AscendC: q_out [T, Nq*D] + k_cache/v_cache (PA_NZ scatter)
  Triton : q/k/v   [T, N*D]  (ND outputs, no paged-cache scatter)

Run:
  python bench_vs_triton.py [--tokens 32768] [--iters 30]
"""

import argparse
import statistics
import sys

import torch
import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401  (registers torch.ops.npu)

from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

D = 128


def make_inputs(tokens, nq, nk, nv, block_size, seed=0, index_mode="contig"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = nq + nk + nv
    qkv = torch.randn(tokens, total * D, generator=g, dtype=torch.float32).to(torch.bfloat16).npu()
    qg = torch.randn(D, generator=g, dtype=torch.float32).to(torch.bfloat16).npu()
    kg = torch.randn(D, generator=g, dtype=torch.float32).to(torch.bfloat16).npu()
    # real call site feeds [T, 1, 1, D] cos/sin to split_qkv_rmsnorm_rope; our op takes [T, D]
    cos = torch.randn(tokens, 1, 1, D, generator=g, dtype=torch.float32).clamp(-1, 1).to(torch.bfloat16).npu()
    sin = torch.randn(tokens, 1, 1, D, generator=g, dtype=torch.float32).clamp(-1, 1).to(torch.bfloat16).npu()
    block_num = (tokens + block_size - 1) // block_size
    if index_mode == "skip":
        index = torch.full((tokens,), -1, dtype=torch.int64, device="npu")
    else:
        index = torch.arange(tokens, dtype=torch.int64, device="npu")
    d1 = D // 16
    q_out = torch.zeros(tokens, nq * D, dtype=torch.bfloat16, device="npu")
    k_cache = torch.zeros(block_num, nk * d1, block_size, 16, dtype=torch.bfloat16, device="npu")
    v_cache = torch.zeros(block_num, nv * d1, block_size, 16, dtype=torch.bfloat16, device="npu")
    return qkv, qg, kg, cos, sin, index, q_out, k_cache, v_cache, block_num


def ulp_stats(got, exp):
    a = got.view(torch.int16).to(torch.int64) & 0xFFFF
    b = exp.view(torch.int16).to(torch.int64) & 0xFFFF
    return (a - b).abs()


def compare(name, got, exp):
    diff = (got.to(torch.float32) - exp.to(torch.float32)).abs()
    ulp = ulp_stats(got, exp)
    hard = ((ulp > 1) & (diff > 1e-5)).sum().item()
    print(
        f"  {name:8s} max_abs={diff.max().item():.3e} "
        f"max_ulp={ulp.max().item()} offenders(>1ulp & >1e-5)={hard}/{ulp.numel()}"
    )


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    lat = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.npu.synchronize()
        lat.append(s.elapsed_time(e))

    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    amortized = s.elapsed_time(e) / iters

    return {
        "median": statistics.median(lat),
        "mean": statistics.mean(lat),
        "min": min(lat),
        "amortized": amortized,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--nq", type=int, default=32)
    ap.add_argument("--nk", type=int, default=8)
    ap.add_argument("--nv", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--index-mode", choices=("contig", "skip"), default="contig",
                    help="skip = index all -1 (AscendC skips the paged-cache scatter)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    T, nq, nk, nv, bs = args.tokens, args.nq, args.nk, args.nv, args.block_size
    qkv, qg, kg, cos4, sin4, index, q_out, k_cache, v_cache, block_num = make_inputs(
        T, nq, nk, nv, bs, index_mode=args.index_mode
    )
    cos, sin = cos4.view(T, D), sin4.view(T, D)
    k_base, v_base = k_cache.clone(), v_cache.clone()
    print(
        f"tokens={T} head_dim={D} heads(q/k/v)=({nq},{nk},{nv}) "
        f"block_size={bs} block_num={block_num} index_mode={args.index_mode}"
    )

    # ---------------- AscendC op ----------------
    def run_ascendc():
        return torch.ops.npu.qkv_rms_norm_rope_cache(
            qkv, qg, kg, cos, sin, index, q_out, k_cache, v_cache, nq, nk, nv, 1e-6
        )

    # ---------------- Triton kernel ----------------
    q_hidden = nq * D
    kv_hidden = nk * D

    def run_triton():
        return split_qkv_rmsnorm_rope(
            qkv,
            sin4,
            cos4,
            q_hidden,
            kv_hidden,
            D,
            eps=1e-6,
            q_weight=qg,
            k_weight=kg,
        )

    # correctness cross-check (same math on the same inputs)
    print("correctness (AscendC vs Triton):")
    qa, ka, va = run_ascendc()
    qt, kt, vt = run_triton()
    torch.npu.synchronize()
    compare("q_out", qa, qt)
    # scatter Triton k/v ND outputs into the same PA_NZ layout for comparison
    d1 = D // 16
    # Build the PA_NZ reference for index == arange(T) (T need not fill block_num*bs).
    # cache layout is [block, head, D/16, block_size, 16] -> flatten (block, offset) as slot.
    k_ref = k_base.clone()
    v_ref = v_base.clone()
    if args.index_mode != "skip":
        pages = torch.arange(T) // bs
        offs = torch.arange(T) % bs
        k_ref.view(block_num, nk, d1, bs, 16)[pages, :, :, offs, :] = kt.view(T, nk, d1, 16)
        v_ref.view(block_num, nv, d1, bs, 16)[pages, :, :, offs, :] = vt.view(T, nv, d1, 16)
    compare("k_cache", k_cache, k_ref)
    compare("v_cache", v_cache, v_ref)

    # ---------------- timing ----------------
    a = bench(run_ascendc, args.warmup, args.iters)
    t = bench(run_triton, args.warmup, args.iters)

    bytes_moved = (
        T * (nq + nk + nv) * D * 2          # qkv read
        + 2 * T * D * 2                     # cos/sin read
        + T * nq * D * 2                    # q_out write
        + T * nk * D * 2                    # k write
        + T * nv * D * 2                    # v write
        + 2 * D * 2                         # gammas
    )

    def line(name, r):
        gbps = bytes_moved / (r["median"] * 1e-3) / 1e9
        print(
            f"{name:34s} median={r['median']*1000:8.1f} us  mean={r['mean']*1000:8.1f} us  "
            f"min={r['min']*1000:8.1f} us  amortized={r['amortized']*1000:8.1f} us  "
            f"traffic={bytes_moved/1e6:6.1f} MB  {gbps:6.1f} GB/s"
        )

    print("timing:")
    line("AscendC qkv_rms_norm_rope_cache", a)
    line("Triton split_qkv_rmsnorm_rope", t)
    print(
        f"speedup (Triton/AscendC, median): {a['median']/t['median']:.2f}x   "
        f"(AscendC/Triton: {t['median']/a['median']:.2f}x)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
