import pytest
import torch
import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401  (registers torch.ops.npu)

HEAD_DIM = 128
D0 = 16          # PA_NZ innermost block (32B for bf16)
D1 = HEAD_DIM // D0


def rms_norm(x, gamma, eps):
    """x: [tokens, heads, D] float32, gamma: [D] float32."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma


def rope(x, cos, sin):
    """Half-and-half RoPE, x: [tokens, heads, D], cos/sin: [tokens, D] float32."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotate_half = torch.cat((-x2, x1), dim=-1)
    return x * cos[:, None, :] + rotate_half * sin[:, None, :]


def scatter_cache(dst, values, index, block_size, num_head):
    """dst: [block_num, num_head, D1, block_size, D0]; values: [tokens, num_head*D]."""
    for t in range(values.shape[0]):
        slot = int(index[t])
        if slot < 0:
            continue
        page, off = divmod(slot, block_size)
        dst[page, :, :, off, :] = values[t].view(num_head, D1, D0).to(dst.dtype)


def golden(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
           num_head_q, num_head_k, num_head_v, eps):
    tokens = qkv.shape[0]
    k_cache = k_cache.clone()
    v_cache = v_cache.clone()

    qkv32 = qkv.to(torch.float32)
    nq, nk, nv = num_head_q, num_head_k, num_head_v
    q, k, v = qkv32.split([nq * HEAD_DIM, nk * HEAD_DIM, nv * HEAD_DIM], dim=-1)
    q = q.reshape(tokens, nq, HEAD_DIM)
    k = k.reshape(tokens, nk, HEAD_DIM)
    v = v.reshape(tokens, nv, HEAD_DIM)

    gq = q_gamma.to(torch.float32)
    gk = k_gamma.to(torch.float32)
    cos32 = cos.to(torch.float32)
    sin32 = sin.to(torch.float32)

    q_rope = rope(rms_norm(q, gq, eps), cos32, sin32).reshape(tokens, nq * HEAD_DIM)
    k_rope = rope(rms_norm(k, gk, eps), cos32, sin32).reshape(tokens, nk * HEAD_DIM)
    v_flat = v.reshape(tokens, nv * HEAD_DIM)

    q_ref = q_rope.to(q_out.dtype)

    k_view = k_cache.view(k_cache.shape[0], nk, D1, k_cache.shape[2], D0)
    v_view = v_cache.view(v_cache.shape[0], nv, D1, v_cache.shape[2], D0)
    scatter_cache(k_view, k_rope, index, k_cache.shape[2], nk)
    scatter_cache(v_view, v_flat, index, v_cache.shape[2], nv)
    return q_ref, k_cache, v_cache


def make_index(tokens, block_num, block_size, skip_every=0):
    slots = block_num * block_size
    assert tokens <= slots
    perm = torch.randperm(slots)[:tokens].to(torch.int64)
    if skip_every:
        perm[::skip_every] = -1
    return perm


def ulp_stats(got, exp):
    """Bit-level ULP distance for bf16 tensors (1 bit step == 1 bf16 ULP)."""
    a = got.view(torch.int16).to(torch.int64) & 0xFFFF
    b = exp.view(torch.int16).to(torch.int64) & 0xFFFF
    d = (a - b).abs()
    # a sign flip around zero is still an adjacent representation; tolerate it as 1 ULP
    return d.max().item(), (d > 1).sum().item(), d.numel()


def check(name, got, exp, max_ulp=1, atol=1e-5):
    """Accept at most 1 bf16 ULP of difference, with a tiny absolute escape hatch for
    values close to zero where the ULP grid is meaningless (e.g. 1.8e-6 vs 1.85e-6)."""
    max_ulp_seen, over, numel = ulp_stats(got, exp)
    diff = (got.to(torch.float32) - exp.to(torch.float32)).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    a = got.view(torch.int16).to(torch.int64) & 0xFFFF
    b = exp.view(torch.int16).to(torch.int64) & 0xFFFF
    ulp = (a - b).abs()
    offenders = ((ulp > max_ulp) & (diff > atol)).sum().item()
    print(f"  {name}: max_abs={max_abs:.3e} max_ulp={max_ulp_seen} offenders={offenders}/{numel}")
    assert offenders == 0, f"{name}: {offenders}/{numel} elements exceed {max_ulp} bf16 ULP and abs {atol}"


@pytest.mark.parametrize(
    "tokens,num_head_q,num_head_k,num_head_v,block_num,block_size,epsilon,skip_every",
    [
        (17, 8, 2, 2, 4, 32, 1e-6, 0),
        (64, 16, 1, 1, 8, 16, 1e-6, 0),
        (33, 4, 2, 2, 4, 16, 1e-5, 3),      # every 3rd token has index == -1
        (1, 2, 1, 1, 2, 16, 1e-6, 0),       # single token edge case
        (128, 32, 8, 8, 8, 32, 1e-6, 0),    # larger head counts
        (2048, 8, 1, 1, 256, 16, 1e-6, 5),  # long sequence: several ub-loop iterations per core
        (1024, 16, 2, 2, 64, 32, 1e-6, 0),
        (96, 1, 1, 1, 8, 16, 1e-6, 0),      # single q head
    ],
)
def test_qkv_rms_norm_rope_cache(tokens, num_head_q, num_head_k, num_head_v,
                                block_num, block_size, epsilon, skip_every):
    torch.manual_seed(0)
    device = "npu"
    dtype = torch.bfloat16
    total_heads = num_head_q + num_head_k + num_head_v

    qkv = torch.randn(tokens, total_heads * HEAD_DIM, dtype=dtype, device=device)
    q_gamma = torch.randn(HEAD_DIM, dtype=dtype, device=device)
    k_gamma = torch.randn(HEAD_DIM, dtype=dtype, device=device)
    cos = torch.randn(tokens, HEAD_DIM, dtype=dtype, device=device).clamp(-1, 1)
    sin = torch.randn(tokens, HEAD_DIM, dtype=dtype, device=device).clamp(-1, 1)
    index = make_index(tokens, block_num, block_size, skip_every)

    q_out = torch.full((tokens, num_head_q * HEAD_DIM), 7.0, dtype=dtype, device=device)
    k_cache = torch.full((block_num, num_head_k * D1, block_size, D0), 3.0, dtype=dtype, device=device)
    v_cache = torch.full((block_num, num_head_v * D1, block_size, D0), -5.0, dtype=dtype, device=device)

    index_npu = index.to(device)
    q_ref, k_ref, v_ref = golden(
        qkv.cpu(), q_gamma.cpu(), k_gamma.cpu(), cos.cpu(), sin.cpu(), index,
        q_out.cpu(), k_cache.cpu(), v_cache.cpu(),
        num_head_q, num_head_k, num_head_v, epsilon)

    q_out_npu, k_cache_npu, v_cache_npu = torch.ops.npu.qkv_rms_norm_rope_cache(
        qkv, q_gamma, k_gamma, cos, sin, index_npu,
        q_out, k_cache, v_cache,
        num_head_q, num_head_k, num_head_v, epsilon)

    print(f"case tokens={tokens} heads=({num_head_q},{num_head_k},{num_head_v}) "
          f"block=({block_num},{block_size}) skip_every={skip_every}")
    # in-place contract: same storage is returned
    assert q_out_npu.data_ptr() == q_out.data_ptr()
    assert k_cache_npu.data_ptr() == k_cache.data_ptr()
    assert v_cache_npu.data_ptr() == v_cache.data_ptr()

    check("q_out", q_out.cpu(), q_ref)
    check("k_cache", k_cache.cpu(), k_ref)
    check("v_cache", v_cache.cpu(), v_ref)


def test_qkv_rms_norm_rope_cache_all_skipped():
    """index == -1 everywhere: outputs must stay byte-identical to their initial content."""
    torch.manual_seed(2)
    tokens, nq, nk, nv, block_num, block_size = 32, 4, 2, 2, 4, 16
    dtype = torch.bfloat16
    qkv = torch.randn(tokens, (nq + nk + nv) * HEAD_DIM, dtype=dtype, device="npu")
    q_gamma = torch.randn(HEAD_DIM, dtype=dtype, device="npu")
    k_gamma = torch.randn(HEAD_DIM, dtype=dtype, device="npu")
    cos = torch.randn(tokens, HEAD_DIM, dtype=dtype, device="npu").clamp(-1, 1)
    sin = torch.randn(tokens, HEAD_DIM, dtype=dtype, device="npu").clamp(-1, 1)
    index = torch.full((tokens,), -1, dtype=torch.int64, device="npu")

    q_out = torch.full((tokens, nq * HEAD_DIM), 7.0, dtype=dtype, device="npu")
    k_cache = torch.full((block_num, nk * D1, block_size, D0), 3.0, dtype=dtype, device="npu")
    v_cache = torch.full((block_num, nv * D1, block_size, D0), -5.0, dtype=dtype, device="npu")
    k_before, v_before = k_cache.clone(), v_cache.clone()

    torch.ops.npu.qkv_rms_norm_rope_cache(
        qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
        nq, nk, nv, 1e-6)

    torch.testing.assert_close(k_cache, k_before, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_before, rtol=0, atol=0)


def test_qkv_rms_norm_rope_cache_deterministic():
    torch.manual_seed(1)
    tokens, nq, nk, nv, block_num, block_size = 40, 8, 2, 2, 4, 32
    dtype = torch.bfloat16
    qkv = torch.randn(tokens, (nq + nk + nv) * HEAD_DIM, dtype=dtype, device="npu")
    q_gamma = torch.randn(HEAD_DIM, dtype=dtype, device="npu")
    k_gamma = torch.randn(HEAD_DIM, dtype=dtype, device="npu")
    cos = torch.randn(tokens, HEAD_DIM, dtype=dtype, device="npu").clamp(-1, 1)
    sin = torch.randn(tokens, HEAD_DIM, dtype=dtype, device="npu").clamp(-1, 1)
    index = torch.randperm(block_num * block_size)[:tokens].to(torch.int64).npu()

    def run():
        q_out = torch.zeros(tokens, nq * HEAD_DIM, dtype=dtype, device="npu")
        k_cache = torch.zeros(block_num, nk * D1, block_size, D0, dtype=dtype, device="npu")
        v_cache = torch.zeros(block_num, nv * D1, block_size, D0, dtype=dtype, device="npu")
        torch.ops.npu.qkv_rms_norm_rope_cache(
            qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache,
            nq, nk, nv, 1e-6)
        return q_out, k_cache, v_cache

    a = run()
    b = run()
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
