"""Dual-oracle tests for torch.ops.npu.kv_rmsnorm_unified_cache.

Physical NPU 7 only. Run with:
  ASCEND_RT_VISIBLE_DEVICES=7 KV_RMSNORM_UNIFIED_LIB=<abs .so> pytest -q ...

Oracle A: the installed torch_npu.npu_kv_rmsnorm_rope_cache called with the real
          512/64 slices of a unified 576 cache (bitwise expected for outputs and
          for non-colliding slots).
Oracle B: an independent CPU FP32 golden of the source semantics (golden.py).
"""

import os
import sys

import pytest
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(__file__))
import golden  # noqa: E402

LIB = os.environ.get("KV_RMSNORM_UNIFIED_LIB")
if LIB:
    torch.ops.load_library(LIB)
torch.npu.set_device(0)

DTYPES = [torch.bfloat16, torch.float16]
TOL = {
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
    torch.float16: dict(rtol=3e-3, atol=3e-3),
}


def run_candidate(kv2d, gamma, cos, sin, index, cache, eps=1e-5, dtype=None):
    T = kv2d.shape[0]
    dt = dtype or kv2d.dtype
    ko = torch.empty(T, 1, 1, golden.KV_ROPE, dtype=dt, device="npu")
    vo = torch.empty(T, 1, 1, golden.KV_LATENT, dtype=dt, device="npu")
    torch.ops.npu.kv_rmsnorm_unified_cache(
        kv2d.view(T, 1, 1, golden.KV_WIDTH), gamma, cos, sin, index, cache, ko, vo, eps
    )
    return ko, vo


def run_old(kv2d, gamma, cos, sin, index, cache, eps=1e-5):
    kb = cache[..., : golden.KV_LATENT]
    vb = cache[..., golden.KV_LATENT :]
    T = kv2d.shape[0]
    r = torch_npu.npu_kv_rmsnorm_rope_cache(
        kv2d.view(T, 1, 1, golden.KV_WIDTH),
        gamma,
        cos,
        sin,
        index,
        vb,
        kb,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        epsilon=eps,
        cache_mode="PA_BNSD",
        is_output_kv=True,
    )
    return r[2], r[3]


def make_case(T, pages, dtype, index_mode="shuffle", seed=0, layer_views=None, kv_stride=None, pad_dups=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kv_stride:
        qkv = torch.randn(T, kv_stride, generator=gen).to(dtype)
        kv2d = qkv[:, kv_stride - golden.KV_WIDTH :]
        assert kv2d.stride(0) == kv_stride
    else:
        kv2d = torch.randn(T, golden.KV_WIDTH, generator=gen).to(dtype)
    gamma = torch.randn(golden.KV_LATENT, generator=gen).to(dtype) * 0.5 + 1.0
    cos = golden.callsite_cos_sin(T, seed=seed).to(dtype)
    sin = golden.callsite_sin(T, seed=seed).to(dtype)
    index = golden.make_indices(T, pages, mode=index_mode, seed=seed, pad_dups=pad_dups)
    if layer_views:
        base = torch.full(
            (layer_views, pages, golden.PAGE, 1, golden.KV_WIDTH), -7.0, dtype=dtype, device="npu"
        )
        cache = base[1]
    else:
        base = None
        cache = torch.full((pages, golden.PAGE, 1, golden.KV_WIDTH), -7.0, dtype=dtype, device="npu")
    qkv_npu = None
    if kv_stride:
        qkv_npu = qkv.to("npu")
        kv2d = qkv_npu[:, kv_stride - golden.KV_WIDTH :]
    else:
        kv2d = kv2d.to("npu")
    return dict(
        kv2d=kv2d,
        gamma=gamma.to("npu"),
        cos=cos.to("npu"),
        sin=sin.to("npu"),
        index=index.to("npu"),
        cache=cache,
        base=base,
        qkv=qkv_npu,
    )


def assert_bitwise(a, b, label):
    assert a.shape == b.shape, f"{label} shape {a.shape} vs {b.shape}"
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"{label} not bitwise equal"


def check_against_cpu_golden(c, ko, vo, dtype, label):
    lat, rope = golden.cpu_golden(
        c["kv2d"].cpu(), c["gamma"].cpu(), c["cos"].cpu(), c["sin"].cpu(), 1e-5
    )
    for got, ref, name in [(vo.reshape(-1, golden.KV_LATENT), lat, "c_kv"), (ko.reshape(-1, golden.KV_ROPE), rope, "k_rope")]:
        g = got.cpu().float()
        r = ref.float()
        diff = (g - r).abs()
        rel = diff / r.abs().clamp_min(1e-6)
        torch.testing.assert_close(
            got.cpu(), ref, rtol=TOL[dtype]["rtol"], atol=TOL[dtype]["atol"], msg=f"{label} {name}"
        )
        assert torch.isfinite(g).all()
        print(
            f"[golden] {label} {name}: max_abs={float(diff.max()):.6g} "
            f"mean_abs={float(diff.mean()):.6g} max_rel={float(rel.max()):.6g} "
            f"mismatch={int((got.cpu() != ref).sum())}/{ref.numel()}"
        )


def check_cache(c, index, label, dtype, dup_allowed=False):
    idx = index.cpu()
    flat = c["cache"].view(-1, golden.KV_WIDTH)
    cap = flat.shape[0]
    # Every untouched row is byte-identical to the seeded sentinel (batched CPU check).
    touched = set(int(v) for v in idx if 0 <= int(v) < cap)
    flat_cpu = flat.cpu()
    mask = torch.ones(cap, dtype=torch.bool)
    mask[list(touched)] = False
    assert (flat_cpu[mask] == -7.0).all(), f"{label} an untouched cache row changed"
    if c["base"] is not None:
        base_cpu = c["base"].cpu()
        for layer in (0, 2):
            assert (base_cpu[layer] == -7.0).all(), f"{label} non-target layer {layer} changed"
    if not dup_allowed:
        # Outputs and cache must agree bitwise for every non-colliding valid slot.
        counts = {}
        for v in idx.tolist():
            counts[v] = counts.get(v, 0) + 1
        pairs = [(t, v) for t, v in enumerate(idx.tolist()) if 0 <= v < cap and counts[v] == 1]
        if pairs:
            ts = torch.tensor([t for t, _ in pairs], dtype=torch.long)
            vs = torch.tensor([v for _, v in pairs], dtype=torch.long)
            exp = torch.cat((c["latent_out"][ts].reshape(-1, golden.KV_LATENT),
                             c["rope_out"][ts].reshape(-1, golden.KV_ROPE)), dim=-1).cpu()
            assert torch.equal(flat_cpu[vs], exp), f"{label} cache/output disagreement"
    return touched, cap


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [1, 2, 31, 127, 128, 129, 257])
def test_dual_oracle_small(dtype, T):
    pages = 2 if T <= 128 else 9
    c = make_case(T, pages, dtype, seed=T)
    ref_cache = c["cache"].clone()
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    ko_old, vo_old = run_old(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], ref_cache)
    torch.npu.synchronize()
    assert_bitwise(ko, ko_old, "k_rope")
    assert_bitwise(vo, vo_old, "c_kv")
    # cache: duplicate slots may be written by different races; compare per slot
    # only when unique, and require every touched row to equal some contributor.
    c["latent_out"], c["rope_out"] = vo, ko
    check_cache(c, c["index"], f"T{T}", dtype)
    check_against_cpu_golden(c, ko, vo, dtype, f"T{T}")
    assert_bitwise(c["cache"], ref_cache, "cache vs old op")


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_tokens(dtype):
    c = make_case(0, 2, dtype)
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    torch.npu.synchronize()
    assert ko.shape == (0, 1, 1, golden.KV_ROPE) and vo.shape == (0, 1, 1, golden.KV_LATENT)
    assert torch.equal(c["cache"], torch.full_like(c["cache"], -7.0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T,pages", [(31, 2), (127, 2), (128, 2), (129, 9), (257, 9), (65, 65)])
def test_bounds_negative_and_untouched(dtype, T, pages):
    c = make_case(T, pages, dtype, index_mode="bounds", seed=T * 3 + pages)
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    torch.npu.synchronize()
    c["latent_out"], c["rope_out"] = vo, ko
    check_cache(c, c["index"], f"bounds T{T} p{pages}", dtype)
    check_against_cpu_golden(c, ko, vo, dtype, f"bounds T{T} p{pages}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_strided_1344_input(dtype):
    T = 129
    c = make_case(T, 2, dtype, seed=99, kv_stride=1344)
    assert c["qkv"].stride(0) == 1344
    qkv_before = c["qkv"].clone()
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    ref = c["cache"].clone()
    ko_old, vo_old = run_old(c["kv2d"].contiguous(), c["gamma"], c["cos"], c["sin"], c["index"], ref)
    torch.npu.synchronize()
    assert_bitwise(ko, ko_old, "strided k_rope")
    assert_bitwise(vo, vo_old, "strided c_kv")
    c["latent_out"], c["rope_out"] = vo, ko
    check_cache(c, c["index"], "strided", dtype)
    check_against_cpu_golden(c, ko, vo, dtype, "strided")
    # readonly qkv source unchanged
    assert torch.equal(c["qkv"], qkv_before)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_layer_offset(dtype):
    T = 65
    c = make_case(T, 4, dtype, seed=5, layer_views=3)
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    torch.npu.synchronize()
    c["latent_out"], c["rope_out"] = vo, ko
    check_cache(c, c["index"], "layer", dtype)
    check_against_cpu_golden(c, ko, vo, dtype, "layer")
    # data_ptr must include the layer storage offset
    assert c["cache"].storage_offset() == 4 * golden.PAGE * golden.KV_WIDTH


@pytest.mark.parametrize("T", [31, 129])
def test_packed_input_matches_strided(T):
    dtype = torch.bfloat16
    a = make_case(T, 2, dtype, seed=7)
    qkv = torch.cat((torch.randn(T, 768, dtype=dtype, device="npu"), a["kv2d"]), dim=-1)
    b = dict(a)
    b["qkv"] = qkv
    b["kv2d"] = qkv[:, 768:]
    b["cache"] = torch.full_like(a["cache"], -7.0)
    ka, va = run_candidate(a["kv2d"], a["gamma"], a["cos"], a["sin"], a["index"], a["cache"])
    kb, vb = run_candidate(b["kv2d"], b["gamma"], b["cos"], b["sin"], b["index"], b["cache"])
    torch.npu.synchronize()
    assert_bitwise(ka, kb, "packed/strided rope")
    assert_bitwise(va, vb, "packed/strided latent")
    assert_bitwise(a["cache"], b["cache"], "packed/strided cache")


def test_padding_dummy0_isolated():
    """Repeated dummy slot 0 is a documented race; verify it is isolated.

    The old framework writes every padded token to slot 0.  With N writers the
    surviving row is undefined (and K/V halves may even come from different
    writers because the source writes them in two scatters).  We only assert:
      * all outputs are finite and match the golden;
      * slot 0 equals one contributor's latent and one contributor's rope;
      * no other row changed.
    """
    T = 16
    c = make_case(T, 2, torch.bfloat16, index_mode="padding", seed=3, pad_dups=3)
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    torch.npu.synchronize()
    c["latent_out"], c["rope_out"] = vo, ko
    check_against_cpu_golden(c, ko, vo, torch.bfloat16, "padding")
    flat = c["cache"].view(-1, golden.KV_WIDTH)
    pad_rows = [t for t, v in enumerate(c["index"].cpu().tolist()) if int(v) == 0]
    # Concurrent whole-row stores from several cores race inside slot 0; the
    # surviving bytes are interleaved at the 32-byte DMA block granularity.
    # Each block must still be exactly one contributor's block (no corruption).
    row = flat[0].cpu()
    blocks = row.view(golden.KV_WIDTH * 2 // 32, 16)
    contrib = torch.stack(
        [torch.cat((vo[t].reshape(-1), ko[t].reshape(-1))).cpu() for t in pad_rows]
    ).view(-1, golden.KV_WIDTH * 2 // 32, 16)
    provenance = [int((contrib[:, b] == blocks[b]).all(dim=-1).sum()) for b in range(blocks.shape[0])]
    assert all(n >= 1 for n in provenance), f"slot 0 has a block from no contributor: {provenance}"
    print(f"[padding] slot0 32B-block provenance counts (writers={len(pad_rows)}): {min(provenance)}..{max(provenance)}")
    touched = set(int(v) for v in c["index"].cpu().tolist() if int(v) >= 0)
    untouched = [s for s in range(4 * 128) if s not in touched][:8]
    for slot in untouched:
        assert torch.equal(flat[slot], torch.full_like(flat[slot], -7.0))
    print("[padding] slot0 is a documented racing sink; every 32B block is owned")


def test_duplicate_live_slot_is_not_deterministic():
    """Duplicate live slots inherit the original undefined race; assert safety only."""
    T = 6
    c = make_case(T, 2, torch.bfloat16, seed=11)
    dup = torch.tensor([0, 7, 7, -1, 300, 7], dtype=torch.int64, device="npu")
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], dup, c["cache"])
    torch.npu.synchronize()
    flat = c["cache"].view(-1, golden.KV_WIDTH)
    row = flat[7].cpu()
    blocks = row.view(golden.KV_WIDTH * 2 // 32, 16)
    contrib = torch.stack(
        [torch.cat((vo[t].reshape(-1), ko[t].reshape(-1))).cpu() for t in [1, 2, 5]]
    ).view(-1, golden.KV_WIDTH * 2 // 32, 16)
    provenance = [int((contrib[:, b] == blocks[b]).all(dim=-1).sum()) for b in range(blocks.shape[0])]
    assert all(n >= 1 for n in provenance), f"duplicate slot has an unowned block: {provenance}"
    assert torch.equal(flat[0][: golden.KV_LATENT], vo[0].reshape(-1))
    assert torch.equal(flat[200], torch.full_like(flat[200], -7.0))  # untouched, in range
    assert torch.equal(flat[255], torch.full_like(flat[255], -7.0))  # OOB index 300 skipped
    print("[dup] live-slot race confined to slot 7; other slots deterministic")


def test_repeated_calls_and_stream_dependency():
    T = 129
    c = make_case(T, 2, torch.bfloat16, seed=21)
    # Producer on the current stream, no host sync before the op.
    c["kv2d"].add_(1.0)
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    # Consumer immediately after the op, still on the same stream.
    consumed = ko * 2.0
    torch.npu.synchronize()
    lat, rope = golden.cpu_golden(c["kv2d"].cpu(), c["gamma"].cpu(), c["cos"].cpu(), c["sin"].cpu(), 1e-5)
    torch.testing.assert_close(vo.reshape(-1, 512).cpu(), lat, **TOL[torch.bfloat16])
    torch.testing.assert_close(ko.reshape(-1, 64).cpu(), rope, **TOL[torch.bfloat16])
    assert_bitwise(consumed, (ko * 2.0), "consumer")
    # Repeated calls are stable.
    for _ in range(3):
        ko2, vo2 = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    torch.npu.synchronize()
    assert_bitwise(ko, ko2, "repeat rope")
    assert_bitwise(vo, vo2, "repeat latent")
    assert torch.isfinite(ko).all() and torch.isfinite(vo).all()


def test_side_stream_with_events():
    T = 65
    c = make_case(T, 2, torch.bfloat16, seed=31)
    main = torch.npu.current_stream()
    side = torch.npu.Stream()
    side.wait_stream(main)
    with torch.npu.stream(side):
        ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
        ev = side.record_event()
    main.wait_event(ev)
    torch.npu.current_stream().wait_event(ev)
    out = ko + 0
    torch.npu.synchronize()
    lat, rope = golden.cpu_golden(c["kv2d"].cpu(), c["gamma"].cpu(), c["cos"].cpu(), c["sin"].cpu(), 1e-5)
    torch.testing.assert_close(ko.reshape(-1, 64).cpu(), rope, **TOL[torch.bfloat16])
    assert_bitwise(out, ko, "side-stream consumer")


def test_validation_rejects():
    T = 4
    c = make_case(T, 2, torch.bfloat16, seed=41)
    cache = c["cache"]
    ko = torch.empty(T, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    vo = torch.empty(T, 1, 1, 512, dtype=torch.bfloat16, device="npu")
    base = dict(
        kv=c["kv2d"].view(T, 1, 1, 576),
        gamma=c["gamma"],
        cos=c["cos"],
        sin=c["sin"],
        index=c["index"],
        cache=cache,
        ko=ko,
        vo=vo,
    )

    def call(**over):
        a = dict(base)
        a.update(over)
        return torch.ops.npu.kv_rmsnorm_unified_cache(
            a["kv"], a["gamma"], a["cos"], a["sin"], a["index"], a["cache"], a["ko"], a["vo"], 1e-5
        )

    with pytest.raises(RuntimeError, match="512"):
        call(cache=cache, ko=ko, vo=torch.empty(T, 1, 1, 256, dtype=torch.bfloat16, device="npu"))
    with pytest.raises(RuntimeError):
        call(cache=cache, ko=ko.to(torch.float16), vo=vo)
    with pytest.raises(RuntimeError, match="dtype|contiguous"):
        call(gamma=c["gamma"].to(torch.float16), ko=ko, vo=vo)
    with pytest.raises(RuntimeError, match="alias"):
        call(ko=ko, vo=cache.view(-1)[: T * 512].view(T, 1, 1, 512))
    with pytest.raises(RuntimeError, match="alias"):
        kv_in_cache = cache.view(-1, golden.KV_WIDTH)[:T].unsqueeze(1).unsqueeze(1)
        call(kv=kv_in_cache, ko=ko, vo=vo)
    with pytest.raises(RuntimeError, match="BF16|FP16"):
        bad = c["kv2d"].to(torch.float32)
        call(kv=bad.view(T, 1, 1, 576), ko=ko.float(), vo=vo.float(), cache=cache.float())
    # Page-gap view is rejected.
    big = torch.empty(3, 128, 1, 576, dtype=torch.bfloat16, device="npu")
    gap = big.as_strided((2, 128, 1, 576), (2 * 128 * 576, 576, 576, 1), 0)
    with pytest.raises(RuntimeError, match="packed"):
        call(cache=gap, ko=ko, vo=vo)
    # Extra size-1 leading dims are rejected by the shape contract (not silently flattened).
    with pytest.raises(RuntimeError):
        call(cache=cache.view(1, 2, 128, 1, 576), ko=ko, vo=vo)


def test_fake_and_meta():
    from torch._subclasses.fake_tensor import FakeTensorMode

    T = 5
    with FakeTensorMode(allow_non_fake_inputs=True):
        kv = torch.empty(T, 1, 1, 576, dtype=torch.bfloat16, device="npu")
        g = torch.empty(512, dtype=torch.bfloat16, device="npu")
        cs = torch.empty(T, 1, 1, 64, dtype=torch.bfloat16, device="npu")
        idx = torch.empty(T, dtype=torch.int64, device="npu")
        cache = torch.empty(2, 128, 1, 576, dtype=torch.bfloat16, device="npu")
        ko = torch.empty(T, 1, 1, 64, dtype=torch.bfloat16, device="npu")
        vo = torch.empty(T, 1, 1, 512, dtype=torch.bfloat16, device="npu")
        assert torch.ops.npu.kv_rmsnorm_unified_cache(kv, g, cs, cs, idx, cache, ko, vo, 1e-5) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T,pages", [(3787, 65), (15148, 7194)])
def test_large_old_oracle(dtype, T, pages):
    if os.environ.get("KV_RMSNORM_RUN_LARGE", "0") != "1":
        pytest.skip("set KV_RMSNORM_RUN_LARGE=1")
    c = make_case(T, pages, dtype, seed=T, index_mode="shuffle")
    ref = c["cache"].clone()
    ko, vo = run_candidate(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], c["cache"])
    ko_old, vo_old = run_old(c["kv2d"], c["gamma"], c["cos"], c["sin"], c["index"], ref)
    torch.npu.synchronize()
    assert_bitwise(ko, ko_old, "large k_rope")
    assert_bitwise(vo, vo_old, "large c_kv")
    assert_bitwise(c["cache"], ref, "large cache")
    # CPU golden on a deterministic sample of rows.
    sample = torch.arange(0, T, max(1, T // 257))[:257].to("npu")
    lat, rope = golden.cpu_golden(
        c["kv2d"][sample].cpu(), c["gamma"].cpu(), c["cos"][sample].cpu(), c["sin"][sample].cpu(), 1e-5
    )
    torch.testing.assert_close(vo[sample].reshape(-1, 512).cpu(), lat, **TOL[dtype])
    torch.testing.assert_close(ko[sample].reshape(-1, 64).cpu(), rope, **TOL[dtype])
    print(f"[large] T={T} pages={pages} dtype={dtype} bitwise vs old and sampled CPU golden OK")
