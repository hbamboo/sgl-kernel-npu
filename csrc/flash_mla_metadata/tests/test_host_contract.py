"""Host input-contract tests: synchronous rejection, empty-optional rules, healthy follow-up."""
import pytest
import torch
import torch_npu

from sgl_kernel_npu import flash_mla_metadata

DEV = "npu"
torch.npu.set_device(torch.npu.current_device())


def meta(seq, cu=None, used=None, heads=20, kv=1, maxq=1, maxkv=128,
         mask=0, left=-1, right=-1, layout="BSND", dqk=576, dv=512):
    return flash_mla_metadata(seq, heads, kv, cu_seqlens_q=cu, seqused_q=used,
                              max_seqlen_q=maxq, max_seqlen_kv=maxkv,
                              head_dim_qk=dqk, head_dim_v=dv,
                              mask_mode=mask, win_left=left, win_right=right,
                              layout_q=layout)


def defined_indices(md):
    """Head8 + FA7 + FD6 defined words at actual per-section offsets; no padding."""
    sec, aic, aiv = int(md[0]), int(md[4]), int(md[5])
    # FaMetadata layout: all FA sections are contiguous first, then all FD
    # sections. Clear() zeroes every record, so only allocation padding beyond
    # 16 + sec*(aic+aiv)*16 is undefined.
    idx = list(range(8))
    fa_base = 16
    fd_base = 16 + sec * aic * 16
    for s in range(sec):
        for c in range(aic):
            e = fa_base + s * aic * 16 + c * 16
            idx.extend(range(e, e + 7))
        for c in range(aiv):
            e = fd_base + s * aiv * 16 + c * 16
            idx.extend(range(e, e + 6))
    return idx


def assert_same(a, b):
    assert int(a[0]) == int(b[0])
    ia, ib = defined_indices(a), defined_indices(b)
    assert len(ia) == len(ib)
    assert torch.equal(a[ia], b[ib])


SEQ = torch.tensor([8, 16], device=DEV, dtype=torch.int32)
CU = torch.tensor([0, 8, 24], device=DEV, dtype=torch.int32)
USED = torch.tensor([8, 16], device=DEV, dtype=torch.int32)


def fresh(seq=SEQ, **kw):
    md = meta(seq, **kw)
    torch.npu.synchronize()
    return md.clone()


BASELINE = fresh()
TND_BASELINE = fresh(layout="TND", cu=CU)


def check_healthy():
    got = fresh()
    assert_same(got, BASELINE)
    got_tnd = fresh(layout="TND", cu=CU)
    assert_same(got_tnd, TND_BASELINE)


INVALID = {
    "nontnd_cu": lambda: meta(SEQ, cu=CU, layout="BSND"),
    "heads_129": lambda: meta(SEQ, heads=129),
    "heads_0": lambda: meta(SEQ, heads=0),
    "bad_layout": lambda: meta(SEQ, layout="XXXX"),
    "mask_5": lambda: meta(SEQ, mask=5),
    "mask_0_window": lambda: meta(SEQ, mask=0, left=5),
    "mask_3_window": lambda: meta(SEQ, mask=3, right=5),
    "win_too_low": lambda: meta(SEQ, mask=4, left=-2),
    "win_too_high": lambda: meta(SEQ, mask=4, left=2**31),
    "kv_heads_2": lambda: meta(SEQ, kv=2),
    "dqk_575": lambda: meta(SEQ, dqk=575),
    "dv_511": lambda: meta(SEQ, dv=511),
    "batch_65536": lambda: meta(torch.ones(65536, device=DEV, dtype=torch.int32)),
    "batch_0": lambda: meta(torch.empty(0, device=DEV, dtype=torch.int32)),
    "tnd_no_cu": lambda: meta(SEQ, layout="TND"),
    "tnd_empty_cu": lambda: meta(SEQ, layout="TND",
                                  cu=torch.empty(0, device=DEV, dtype=torch.int32)),
    "tnd_rank0_cu": lambda: meta(SEQ, layout="TND",
                                  cu=torch.tensor(0, device=DEV, dtype=torch.int32)),
    "tnd_cu_short": lambda: meta(SEQ, layout="TND",
                                  cu=torch.tensor([0, 8], device=DEV, dtype=torch.int32)),
    "used_short": lambda: meta(SEQ, used=torch.tensor([8], device=DEV, dtype=torch.int32)),
    "cu_rank2": lambda: meta(SEQ, layout="TND",
                             cu=torch.tensor([[0], [8], [24]], device=DEV, dtype=torch.int32)),
    "cu_dtype": lambda: meta(SEQ, layout="TND",
                              cu=CU.to(torch.float32)),
    "used_dtype": lambda: meta(SEQ, used=USED.to(torch.float32)),
    "cu_cpu": lambda: meta(SEQ, layout="TND",
                            cu=torch.tensor([0, 8, 24], dtype=torch.int32)),
    "used_noncontiguous": lambda: meta(
        SEQ, used=torch.arange(6, device=DEV, dtype=torch.int32)[::2]),
    "bsnd_empty_cu_float": lambda: meta(SEQ,
        cu=torch.empty(0, device=DEV, dtype=torch.float32)),
    "bsnd_empty_used_cpu": lambda: meta(SEQ,
        used=torch.empty(0, dtype=torch.int32)),
    "cu_rank0": lambda: meta(SEQ, cu=torch.tensor(7, device=DEV, dtype=torch.int32)),
    "used_rank0": lambda: meta(SEQ, used=torch.tensor(7, device=DEV, dtype=torch.int32)),
    "cu_empty_2d": lambda: meta(SEQ,
        cu=torch.empty(0, 3, device=DEV, dtype=torch.int32)),
    "used_empty_2d": lambda: meta(SEQ,
        used=torch.empty(0, 3, device=DEV, dtype=torch.int32)),
    "maxq_minus2": lambda: meta(SEQ, maxq=-2),
    "maxkv_minus2": lambda: meta(SEQ, maxkv=-2),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_host_rejects_then_context_healthy(name):
    with pytest.raises(RuntimeError):
        INVALID[name]()
    check_healthy()


def test_empty_optionals_mean_omitted():
    empty = torch.empty(0, device=DEV, dtype=torch.int32)
    assert_same(fresh(cu=empty), BASELINE)
    assert_same(fresh(used=empty), BASELINE)
    assert_same(fresh(cu=empty, used=empty), BASELINE)
    assert_same(fresh(layout="TND", cu=CU, used=empty), TND_BASELINE)
