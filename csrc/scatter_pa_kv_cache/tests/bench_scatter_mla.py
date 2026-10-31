"""Exploratory device-event timing; formal profiler validation is separate."""
import os,json,statistics,time,itertools
from pathlib import Path
import torch
import torch_npu
torch.set_num_threads(4)
torch.ops.load_library(os.environ['SGL_SCATTER_MLA_TEST_LIBRARY'])
torch.npu.set_device(0)
out=Path(os.environ.get('SCATTER_BENCH_OUT','scatter-results/explore7.json'))
torch.manual_seed(712)
results=[]
warm=torch.randn(1024,device='npu')

def measure(fn):
    for _ in range(3):fn()
    torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):fn()
    samples=[]
    for _ in range(10):
        torch.argmax(warm)
        a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)*1000)
    return statistics.median(samples),samples

for n,layout in itertools.product([33305],['sequential','shuffled_pages','random_slots']):
    h,nd,rd,vd,cap,page=20,192,64,256,49152,128
    x=torch.randn(n,h,nd+vd,dtype=torch.bfloat16,device='npu')
    rope=torch.randn(n,1,rd,dtype=torch.bfloat16,device='npu')
    kc=torch.empty(cap//page,page,h,nd+rd,dtype=x.dtype,device='npu');vc=torch.empty_like(kc)
    if layout=='sequential':s=torch.arange(n,dtype=torch.int64)
    elif layout=='shuffled_pages':s=(torch.randperm(cap//page)[:,None]*page+torch.arange(page)[None,:]).flatten()[:n]
    else:s=torch.randperm(cap)[:n]
    slots=s.npu();indices=slots.view(-1,1)
    k=torch.cat([x[...,:nd],rope.expand(-1,h,-1)],dim=-1);v=x[...,nd:].contiguous()
    def old_scatter():
        torch_npu.npu_scatter_nd_update_(kc.view(cap,h,256),indices,k)
        torch_npu.npu_scatter_nd_update_(vc.view(cap,h,256),indices,v)
    def old_full():
        kk=torch.cat([x[...,:nd],rope.expand(-1,h,-1)],dim=-1)
        vv=x[...,nd:].contiguous()
        torch_npu.npu_scatter_nd_update_(kc.view(cap,h,256),indices,kk)
        torch_npu.npu_scatter_nd_update_(vc.view(cap,h,256),indices,vv)
    for name,fn in [('old_two_scatter',old_scatter),('old_layout_and_scatter',old_full)]:
        us,samples=measure(fn)
        row=dict(n=n,layout=layout,name=name,median_us=us,samples_us=samples)
        results.append(row);out.write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
    for tile,cores,coalesce in itertools.product([1,2,4,6],[16,32,48,64],[False,True]):
        fn=lambda:torch.ops.npu.scatter_pa_kv_cache_mla(x,rope,slots,kc,vc,nd,tile,cores,coalesce)
        try:us,samples=measure(fn)
        except RuntimeError as e:
            if 'UB capacity' in str(e):continue
            raise
        minimum=n*(h*(nd+vd)*2+rd*2+h*(nd+rd+vd)*2)
        row=dict(n=n,layout=layout,name='fused',tile=tile,cores=cores,coalesce=coalesce,median_us=us,samples_us=samples,effective_TBps=minimum/us/1e6)
        results.append(row);out.write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
    del x,rope,kc,vc,k,v,slots,indices
    torch.npu.empty_cache()

# Large contiguous copy ceiling: read+write bytes, exceeds device L2 capacity.
src=torch.empty(512*1024*1024,dtype=torch.bfloat16,device='npu');dst=torch.empty_like(src)
us,samples=measure(lambda:dst.copy_(src))
row=dict(name='contiguous_copy_ceiling',bytes=src.numel()*src.element_size()*2,median_us=us,samples_us=samples,effective_TBps=src.numel()*src.element_size()*2/us/1e6)
results.append(row);out.write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
