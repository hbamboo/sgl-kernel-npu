"""Vendor-free FlashMLA metadata tests: layout invariants, graph replay, errors."""
import os

import pytest
import torch
import torch_npu

from sgl_kernel_npu import flash_mla, flash_mla_metadata

DEV = "npu"
torch.npu.set_device(torch.npu.current_device())


def valid_words(md):
    sec, aic, aiv = int(md[0]), int(md[4]), int(md[5])
    return 16 + sec * (aic + aiv) * 16


def assert_structure(md):
    assert md.dtype == torch.int32
    assert md.dim() == 1 and md.numel() >= 16
    sec, is_fd, mbase, s2base, aic, aiv, out_layout = [int(md[i]) for i in range(7)]
    assert sec >= 1
    assert is_fd in (0, 1)
    assert mbase == 96 and s2base == 112
    assert aic > 0 and aiv == 2 * aic
    assert out_layout == 0
    assert md.numel() >= valid_words(md)
    stride = aic * 16 + aiv * 16
    for s in range(sec):
        base = 16 + s * stride
        fa_start = base
        for c in range(aic):
            e = fa_start + c * 16
            # FA range is lexicographic (bn2, m, s2); inner coordinates reset
            # when the range crosses batch/head or M boundaries.
            start = (int(md[e + 0]), int(md[e + 1]), int(md[e + 2]))
            end = (int(md[e + 3]), int(md[e + 4]), int(md[e + 5]))
            assert end >= start
        fd_start = fa_start + aic * 16
        for c in range(aiv):
            e = fd_start + c * 16
            assert int(md[e + 3]) >= 0  # workspace num
            assert int(md[e + 4]) >= 0  # m start
            assert int(md[e + 5]) >= 0  # m len


def meta(cache, heads=20, maxq=1, maxkv=128, layout="BSND", mask=0,
         left=-1, right=-1, cu=None, used=None):
    seq = torch.tensor(cache, device=DEV, dtype=torch.int32)
    cu_t = torch.tensor(cu, device=DEV, dtype=torch.int32) if cu is not None else None
    used_t = torch.tensor(used, device=DEV, dtype=torch.int32) if used is not None else None
    md = flash_mla_metadata(seq, heads, 1, cu_seqlens_q=cu_t, seqused_q=used_t,
                            max_seqlen_q=maxq, max_seqlen_kv=maxkv,
                            mask_mode=mask, win_left=left, win_right=right,
                            layout_q=layout)
    torch.npu.synchronize()
    return md


@pytest.mark.parametrize("layout,mask", [("BSND", 0), ("BNSD", 0), ("TND", 0),
                                         ("BSND", 3), ("BSND", 4)])
def test_layouts_and_masks(layout, mask):
    cu = [0, 1, 3, 6] if layout == "TND" else None
    md = meta([17, 129, 3014], heads=20, maxq=4, maxkv=3014, layout=layout,
              mask=mask, left=3 if mask == 4 else -1, right=5 if mask == 4 else -1,
              cu=cu, used=[1, 2, 3] if layout == "TND" else None)
    assert_structure(md)


def test_multi_section_and_fd_states():
    md_long = meta([32768, 30000], heads=32, maxq=32, maxkv=32768)
    assert_structure(md_long)
    assert int(md_long[0]) >= 2, "long prefill must exercise multiple sections"
    md_short = meta([1, 16, 111], heads=128, maxq=1, maxkv=111)
    assert_structure(md_short)


def test_graph_replay_rewrites_metadata():
    seq = torch.tensor([8, 16], device=DEV, dtype=torch.int32)
    expected = {}
    for tag, vals in (("short", [8, 16]), ("long", [4096, 30000])):
        seq.copy_(torch.tensor(vals, device=DEV, dtype=torch.int32))
        expected[tag] = meta(vals, heads=20, maxq=1, maxkv=32768).clone()

    seq.fill_(8)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        md_g = flash_mla_metadata(seq, 20, 1, max_seqlen_q=1, max_seqlen_kv=32768)
    torch.npu.synchronize()
    md_g.fill_(0)
    torch.npu.synchronize()
    for tag, vals in (("long", [4096, 30000]), ("short", [8, 16]), ("long", [4096, 30000])):
        seq.copy_(torch.tensor(vals, device=DEV, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        v = valid_words(expected[tag])
        assert torch.equal(md_g[:v], expected[tag][:v]), f"replay mismatch for {tag}"


def test_compute_graph_short_long_short():
    torch.manual_seed(11)
    batch, heads, page, maxkv = 1, 20, 64, 4096
    blocks = maxkv // page
    q = torch.randn(batch, 1, heads, 576, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(batch * blocks, page, 1, 576, device=DEV, dtype=torch.bfloat16)
    bt = torch.arange(batch * blocks, device=DEV, dtype=torch.int32).reshape(batch, blocks)
    seq = torch.tensor([64], device=DEV, dtype=torch.int32)
    srcs = {"short": 64, "long": 4096}
    expected = {}
    for tag, length in srcs.items():
        seq.fill_(length)
        expected[tag] = flash_mla(q, k, block_table=bt, cache_seqlens=seq,
                                  max_seqlen_q=1, max_seqlen_kv=maxkv,
                                  softmax_scale=0.0625)[0].clone()
    seq.fill_(64)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_g = flash_mla(q, k, block_table=bt, cache_seqlens=seq,
                          max_seqlen_q=1, max_seqlen_kv=maxkv, softmax_scale=0.0625)[0]
    torch.npu.synchronize()
    for tag, length in (("long", 4096), ("short", 64), ("long", 4096)):
        seq.fill_(length)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(out_g, expected[tag]), f"compute replay mismatch for {tag}"


def test_shape_errors_are_rejected():
    seq = torch.tensor([8, 16], device=DEV, dtype=torch.int32)
    with pytest.raises(RuntimeError):
        flash_mla_metadata(seq, 20, 1,
                           cu_seqlens_q=torch.tensor([0, 8], device=DEV, dtype=torch.int32),
                           max_seqlen_q=1, max_seqlen_kv=128, layout_q="TND")
    with pytest.raises(RuntimeError):
        flash_mla_metadata(seq, 20, 1,
                           cu_seqlens_q=torch.tensor([0, 8, 24], device=DEV, dtype=torch.int32),
                           seqused_q=torch.tensor([8], device=DEV, dtype=torch.int32),
                           max_seqlen_q=1, max_seqlen_kv=128, layout_q="TND")
    with pytest.raises(RuntimeError):
        flash_mla_metadata(seq, 20, 1, max_seqlen_q=1, max_seqlen_kv=128, layout_q="TND")


def test_fake_tensor_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        seq = torch.empty(2, device=DEV, dtype=torch.int32)
        md = torch.ops.npu.flash_mla_metadata(seq, 20, 1, None, None, 1, 128, 576, 512, 0, -1, -1, "BSND")
        assert md.dtype == torch.int32 and md.numel() > 0
