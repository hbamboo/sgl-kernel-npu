"""Fresh-cache replay, stream ownership, and nonstandard stride fallback."""
import os
from pathlib import Path
exec(Path(__file__).with_name('test_packed_query.py').read_text().split('rows=[]')[0])
assert C.mla_prolog_v3_small_decode_version >= 2
for direct_cache in [False,True]:
 for t in range(1,11):
    args,kw,_=make(t,direct_cache,16 if t%2 else 128)
    initial=[args[9].clone(),args[10].clone()]
    out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=True)
    graph=torch.npu.NPUGraph()
    C.mla_prolog_v3_out(*args,*out,**kw);torch.npu.synchronize()
    with torch.npu.graph(graph):C.mla_prolog_v3_out(*args,*out,**kw)
    for rep in range(3):
        args[0].mul_(0.9);kw['cache_index'].copy_(kw['cache_index'].roll(1))
        args[9].copy_(initial[0]);args[10].copy_(initial[1])
        ref=C.mla_prolog_v3(*args,**kw);expected=[args[9].clone(),args[10].clone()]
        args[9].copy_(initial[0]);args[10].copy_(initial[1])
        out[0].fill_(-19);out[1].fill_(-19);out[3].fill_(-19)
        graph.replay();torch.npu.synchronize()
        assert all(torch.equal(x,y) or x.numel()==y.numel()==0 for x,y in zip(ref,out)),(t,direct_cache,rep)
        assert torch.equal(args[9],expected[0]) and torch.equal(args[10],expected[1]),('cache',t,direct_cache,rep)
    print('PASS refreshed capture',t,direct_cache,flush=True)

# Distinct streams keep independent workspace and cache contents.
items=[]
for t in [1,9]:
    args,kw,_=make(t,True);initial=[args[9].clone(),args[10].clone()]
    ref=C.mla_prolog_v3(*args,**kw);expected=[args[9].clone(),args[10].clone()]
    args[9].copy_(initial[0]);args[10].copy_(initial[1])
    out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=True)
    items.append((args,kw,out,ref,expected,torch.npu.Stream()))
torch.npu.synchronize()
for args,kw,out,ref,expected,stream in items:
    with torch.npu.stream(stream):
        for _ in range(5):C.mla_prolog_v3_out(*args,*out,**kw)
torch.npu.synchronize()
for args,kw,out,ref,expected,stream in items:
    assert all(torch.equal(x,y) or x.numel()==y.numel()==0 for x,y in zip(ref,out))
    assert all(torch.equal(x,y) for x,y in zip(args[9:11],expected))
print('PASS mixed decode multi-stream',flush=True)

# Preserve the existing aligned regular-stride out API when stride is not576.
for t in [1,10]:
    args,kw,_=make(t,True);ref=C.mla_prolog_v3(*args,**kw)
    out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True)
    backing=torch.full((t,20,608),-19,device=dev,dtype=torch.bfloat16)
    out=(backing[...,:512],backing[...,512:576],*out[2:])
    C.mla_prolog_v3_out(*args,*out,**kw);torch.npu.synchronize()
    assert all(torch.equal(x,y) or x.numel()==y.numel()==0 for x,y in zip(ref,out))
    assert torch.all(backing[...,576:]==-19)
print('PASS nonstandard stride fallback',flush=True)
