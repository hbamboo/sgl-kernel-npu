"""CPU byte-exact golden for fused MLA cache assembly, physical NPU 7 only."""
import os
import pytest
import torch
import torch_npu

if os.environ.get('SGL_SCATTER_MLA_TEST_LIBRARY'):
    torch.ops.load_library(os.environ['SGL_SCATTER_MLA_TEST_LIBRARY'])
else:
    import sgl_kernel_npu
torch.npu.set_device(0)

def call(x,r,s,k,v,**kw):
    torch.ops.npu.scatter_pa_kv_cache_mla(x,r,s,k,v,**kw)

def bits(t):return t.cpu().contiguous().view(torch.uint8)

def tensors(n,h=20,nd=192,rd=64,vd=256,dtype=torch.bfloat16,idx=torch.int64,random=True):
    cap=((n+255)//128)*128
    x=torch.randn(n,h,nd+vd).to(dtype)
    r=torch.randn(n,1,rd).to(dtype)
    s=(torch.randperm(cap)[:n] if random else torch.arange(n)).to(idx)
    k=torch.full((cap//128,128,h,nd+rd),-3,dtype=dtype)
    v=torch.full((cap//128,128,h,vd),5,dtype=dtype)
    ek,ev=k.clone(),v.clone()
    ek.view(cap,h,nd+rd)[s.long()]=torch.cat((x[...,:nd],r.expand(-1,h,-1)),dim=-1)
    ev.view(cap,h,vd)[s.long()]=x[...,nd:]
    return [t.npu() for t in [x,r,s,k,v]],ek,ev

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
@pytest.mark.parametrize('n',[1,3,63,65,127,129,513,33305])
@pytest.mark.parametrize('random',[False,True])
def test_actual_geometry(dtype,idx,n,random):
    ts,ek,ev=tensors(n,dtype=dtype,idx=idx,random=random)
    call(*ts)
    assert torch.equal(bits(ts[3]),bits(ek))
    assert torch.equal(bits(ts[4]),bits(ev))

@pytest.mark.parametrize('tile',[1,2,3,4])
@pytest.mark.parametrize('cores',[1,16,32,48,64])
@pytest.mark.parametrize('coalesce',[False,True])
def test_tiling(tile,cores,coalesce):
    ts,ek,ev=tensors(517)
    call(*ts,tile_tokens=tile,core_num=cores,coalesce=coalesce)
    assert torch.equal(bits(ts[3]),bits(ek))
    assert torch.equal(bits(ts[4]),bits(ev))

def test_stream_and_graph():
    ts,ek,ev=tensors(129)
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        call(*ts)
    stream.synchronize()
    assert torch.equal(bits(ts[3]),bits(ek))
    for _ in range(3):call(*ts)
    torch.npu.synchronize()
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):call(*ts)
    ts[3].fill_(-3);ts[4].fill_(5)
    g.replay();torch.npu.synchronize()
    assert torch.equal(bits(ts[3]),bits(ek))
    assert torch.equal(bits(ts[4]),bits(ev))

def test_invalid_padding_slots():
    ts,_,_=tensors(3)
    ts[2].copy_(torch.tensor([-1,7,ts[3].shape[0]*128],device='npu'))
    call(*ts)
    k=ts[3].cpu().view(-1,20,256)
    assert (k[0]==-3).all()
    assert torch.equal(k[7,:,:192],ts[0][1,:,:192].cpu())

def test_fake():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        ts=[torch.empty(129,20,448,device='npu',dtype=torch.bfloat16),torch.empty(129,1,64,device='npu',dtype=torch.bfloat16),torch.empty(129,device='npu',dtype=torch.int64),torch.empty(2,128,20,256,device='npu',dtype=torch.bfloat16),torch.empty(2,128,20,256,device='npu',dtype=torch.bfloat16)]
        assert call(*ts) is None


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('idx',[torch.int32,torch.int64])
@pytest.mark.parametrize('n',[0,1,65,129,8192,30080])
@pytest.mark.parametrize('random',[False,True])
def test_dense_outputs_and_cache(dtype,idx,n,random):
    ts,ek,ev=tensors(n,dtype=dtype,idx=idx,random=random)
    x,r,s,k,v=ts
    ko=torch.empty(n,20,256,dtype=dtype,device='npu')
    vo=torch.empty_like(ko)
    call(*ts,key_out=ko,value_out=vo)
    assert torch.equal(bits(ko),bits(torch.cat((x[...,:192],r.expand(-1,20,-1)),dim=-1)))
    assert torch.equal(bits(vo),bits(x[...,192:]))
    assert torch.equal(bits(k),bits(ek))
    assert torch.equal(bits(v),bits(ev))


def test_dense_invalid_slots_still_produce_fia_rows():
    ts,_,_=tensors(3)
    cap=ts[3].shape[0]*128
    ts[2].copy_(torch.tensor([-1,7,cap],dtype=ts[2].dtype,device='npu'))
    ko=torch.empty(3,20,256,dtype=ts[0].dtype,device='npu');vo=torch.empty_like(ko)
    call(*ts,key_out=ko,value_out=vo)
    assert torch.equal(bits(ko),bits(torch.cat((ts[0][...,:192],ts[1].expand(-1,20,-1)),dim=-1)))
    assert torch.equal(bits(vo),bits(ts[0][...,192:]))
    expected_k=torch.full_like(ts[3],-3);expected_v=torch.full_like(ts[4],5)
    expected_k.view(cap,20,256)[7]=ko[1]
    expected_v.view(cap,20,256)[7]=vo[1]
    assert torch.equal(bits(ts[3]),bits(expected_k))
    assert torch.equal(bits(ts[4]),bits(expected_v))


def test_dense_stream_and_graph_replay_updates_both_outputs():
    ts,_,_=tensors(129)
    ko=torch.empty(129,20,256,dtype=ts[0].dtype,device='npu');vo=torch.empty_like(ko)
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):call(*ts,key_out=ko,value_out=vo)
    stream.synchronize()
    for _ in range(3):call(*ts,key_out=ko,value_out=vo)
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):call(*ts,key_out=ko,value_out=vo)
    ts[0].copy_(torch.randn_like(ts[0]));ts[1].copy_(torch.randn_like(ts[1]))
    ko.fill_(-9);vo.fill_(-9);ts[3].fill_(-3);ts[4].fill_(5)
    graph.replay();torch.npu.synchronize()
    expected=torch.cat((ts[0][...,:192],ts[1].expand(-1,20,-1)),dim=-1)
    assert torch.equal(bits(ko),bits(expected))
    assert torch.equal(bits(vo),bits(ts[0][...,192:]))
    slots=ts[2].cpu().long()
    assert torch.equal(bits(ts[3].view(-1,20,256)[ts[2].long()]),bits(expected))
    assert torch.equal(bits(ts[4].view(-1,20,256)[ts[2].long()]),bits(vo))


def test_dense_validation_and_fake():
    ts,_,_=tensors(3)
    ko=torch.empty(3,20,256,dtype=ts[0].dtype,device='npu');vo=torch.empty_like(ko)
    with pytest.raises(RuntimeError,match='supplied together'):call(*ts,key_out=ko)
    with pytest.raises(RuntimeError,match='alias each other'):call(*ts,key_out=ko,value_out=ko)
    with pytest.raises(RuntimeError,match='alias inputs or caches'):
        call(*ts,key_out=ts[3].view(-1,20,256)[:3],value_out=vo)
    with pytest.raises(RuntimeError,match='key_out must'):
        call(*ts,key_out=ko[:2],value_out=vo)
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs=True):
        assert call(*ts,key_out=ko,value_out=vo) is None
