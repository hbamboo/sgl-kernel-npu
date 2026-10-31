"""Bitwise cache semantics, layouts, streams, and graph tests for the port."""
import os
from pathlib import Path
import pytest
import torch
import torch_npu
torch.set_num_threads(4)

LIB=os.environ.get('SGL_SCATTER_TEST_LIBRARY')
if LIB:
    torch.ops.load_library(LIB)
else:
    import sgl_kernel_npu

torch.npu.set_device(int(os.environ.get('SCATTER_DEVICE','0')))

def call(k,kc,s,v,vc,**kw):
    return torch.ops.npu.scatter_pa_kv_cache(k,kc,s,v,vc,**kw)

def bits(t):
    return t.cpu().contiguous().view(torch.uint8)

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32,torch.int8,torch.int16,torch.int32,torch.uint8,torch.uint16,torch.uint32,torch.float8_e4m3fn,torch.float8_e5m2])
@pytest.mark.parametrize('idx_dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('tokens,heads,kdim,vdim',[(1,1,16,16),(127,3,32,64),(129,20,192,256),(513,2,17,31)])
def test_normal(dtype,idx_dtype,tokens,heads,kdim,vdim):
    capacity=768;page=128
    k=torch.randn(tokens,heads,kdim).to(dtype)
    v=torch.randn(tokens,heads,vdim).to(dtype)
    kc=torch.full((capacity//page,page,heads,kdim),3,dtype=dtype)
    vc=torch.full((capacity//page,page,heads,vdim),5,dtype=dtype)
    slots=torch.randperm(capacity)[:tokens].to(idx_dtype)
    expected_k=kc.clone();expected_v=vc.clone()
    expected_k.view(capacity,heads,kdim).view(torch.uint8).index_copy_(0,slots.long(),k.view(torch.uint8))
    expected_v.view(capacity,heads,vdim).view(torch.uint8).index_copy_(0,slots.long(),v.view(torch.uint8))
    kn,vn,kcn,vcn,sn=[x.npu() for x in [k,v,kc,vc,slots]]
    call(kn,kcn,sn,vn,vcn)
    assert torch.equal(bits(kcn),bits(expected_k))
    assert torch.equal(bits(vcn),bits(expected_v))

@pytest.mark.parametrize('idx_dtype',[torch.int32,torch.int64])
def test_noncontiguous_inputs_and_cache(idx_dtype):
    n,h,d=193,5,32;capacity=512;page=128
    src=torch.randn(n,h,2*d,dtype=torch.bfloat16)
    key=src[...,:d];value=src[...,d:]
    raw=torch.full((capacity//page,page,h,2*d),7,dtype=torch.bfloat16)
    expected=raw.clone();slots=torch.randperm(capacity)[:n].to(idx_dtype)
    expected.view(capacity,h,2*d)[slots.long(),:,:d]=key
    expected.view(capacity,h,2*d)[slots.long(),:,d:]=value
    xn=src.npu();cn=raw.npu();sn=slots.npu()
    call(xn[...,:d],cn[...,:d],sn,xn[...,d:],cn[...,d:])
    assert torch.equal(bits(cn),bits(expected))

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16,torch.float32,torch.int8])
@pytest.mark.parametrize('idx_dtype',[torch.int32,torch.int64])
def test_nz(dtype,idx_dtype):
    n,h,kd,vd=193,3,32,64;cap=512;page=128
    unit=32//torch.empty((),dtype=dtype).element_size()
    k=torch.randn(n,h,kd).to(dtype);v=torch.randn(n,h,vd).to(dtype)
    slots=torch.randperm(cap)[:n].to(idx_dtype)
    kc=torch.full((cap//page,h*kd//unit,page,unit),3,dtype=dtype)
    vc=torch.full((cap//page,h*vd//unit,page,unit),5,dtype=dtype)
    ek,ev=kc.clone(),vc.clone()
    for i,slot in enumerate(slots.tolist()):
        ek[slot//page,:,slot%page,:]=k[i].reshape(-1,unit)
        ev[slot//page,:,slot%page,:]=v[i].reshape(-1,unit)
    kn,vn,kcn,vcn,sn=[x.npu() for x in [k,v,kc,vc,slots]]
    call(kn,kcn,sn,vn,vcn,cache_mode='PA_NZ')
    assert torch.equal(bits(kcn),bits(ek))
    assert torch.equal(bits(vcn),bits(ev))

def test_padding_slots_are_skipped():
    k=torch.randn(4,1,32,dtype=torch.bfloat16).npu()
    v=(k*2).contiguous();kc=torch.zeros(2,128,1,32,dtype=torch.bfloat16).npu();vc=kc.clone()
    slots=torch.tensor([-1,4,256,9],dtype=torch.int64).npu()
    call(k,kc,slots,v,vc)
    assert torch.equal(kc.view(256,1,32)[4].cpu(),k[1].cpu())
    assert torch.equal(kc.view(256,1,32)[9].cpu(),k[3].cpu())
    assert kc.view(256,1,32)[0].count_nonzero().item()==0

def test_empty():
    k=torch.empty(0,2,32,dtype=torch.bfloat16).npu();c=torch.ones(1,128,2,32,dtype=torch.bfloat16).npu()
    call(k,c,torch.empty(0,dtype=torch.int64).npu(),k,c.clone())
    assert torch.all(c==1).item()

def test_nondefault_stream():
    s=torch.npu.Stream()
    with torch.npu.stream(s):
        k=torch.randn(257,2,32,dtype=torch.bfloat16).npu()
        v=k*2;kc=torch.zeros(4,128,2,32,dtype=torch.bfloat16).npu();vc=kc.clone()
        slots=torch.arange(257,dtype=torch.int64).npu()
        call(k,kc,slots,v,vc)
        out=kc.view(-1,2,32)[:257].clone()
    s.synchronize()
    assert torch.equal(out.cpu(),k.cpu())

def test_graph_replay():
    k=torch.randn(129,2,32,dtype=torch.bfloat16).npu();v=k.clone()
    kc=torch.zeros(2,128,2,32,dtype=torch.bfloat16).npu();vc=kc.clone();slots=torch.arange(129,dtype=torch.int64).npu()
    for _ in range(3):call(k,kc,slots,v,vc)
    torch.npu.synchronize()
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):call(k,kc,slots,v,vc)
    k.fill_(2);v.fill_(4);g.replay();torch.npu.synchronize()
    assert torch.equal(kc.view(-1,2,32)[:129].cpu(),k.cpu())
    assert torch.equal(vc.view(-1,2,32)[:129].cpu(),v.cpu())

def test_fake_tensor_registration():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        k=torch.empty(129,2,32,device='npu',dtype=torch.bfloat16)
        c=torch.empty(2,128,2,32,device='npu',dtype=torch.bfloat16)
        slots=torch.empty(129,device='npu',dtype=torch.int64)
        assert call(k,c,slots,k,c) is None
