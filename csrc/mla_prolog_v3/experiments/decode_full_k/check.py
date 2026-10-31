"""Decode-only packed Query correctness, capture, guards and paired timings.

Run with source-tree PYTHONPATH and ASCEND_RT_VISIBLE_DEVICES set to an idle card.
The reference is the same kernel's contiguous-output path (strict equality).
"""
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from sgl_kernel_npu.mla_prolog_v3 import allocate_outputs
from sgl_kernel_npu.mla_prolog_v3_compat import direct

torch.manual_seed(42)
torch.npu.set_device(0)
dev='npu:0'
C=direct()
assert C.mla_prolog_v3_packed_query_version >= 1
spec=importlib.util.spec_from_file_location('packed_adapter', '/sgl-workspace/sglang-p-kv-opt/python/sglang/srt/hardware_backend/npu/attention/flash_mla_adapter.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
nz=lambda x:torch_npu.npu_format_cast(x,29)
wdq=nz(torch.randn(2048,768,device=dev,dtype=torch.bfloat16)*0.02)
wuq=nz(torch.randn(768,5120,device=dev,dtype=torch.bfloat16)*0.02)
wuk=torch.randn(20,192,512,device=dev,dtype=torch.bfloat16)*0.02
wdkv=nz(torch.randn(2048,576,device=dev,dtype=torch.bfloat16)*0.02)
gq=torch.rand(768,device=dev,dtype=torch.bfloat16)+0.5
gkv=torch.rand(512,device=dev,dtype=torch.bfloat16)+0.5

def make(t, direct_cache=False, page=128):
 x=torch.randn(t,2048,device=dev,dtype=torch.bfloat16)*0.1
 angle=torch.randn(t,32)
 sin=angle.sin().repeat(1,2).to(device=dev,dtype=torch.bfloat16)
 cos=angle.cos().repeat(1,2).to(device=dev,dtype=torch.bfloat16)
 blocks=(2*t+page-1)//page+2
 idx=(torch.randperm(2*t)[:t]+page).to(device=dev,dtype=torch.int64)
 full=torch.full((blocks,page,1,576),-13,device=dev,dtype=torch.bfloat16)
 if direct_cache:kv,kr=full[...,:512],full[...,512:]
 else:
  kv=torch.full((blocks,page,1,512),-13,device=dev,dtype=torch.bfloat16)
  kr=torch.full((blocks,page,1,64),-13,device=dev,dtype=torch.bfloat16)
 return [x,wdq,wuq,wuk,wdkv,gq,gkv,sin,cos,kv,kr],dict(cache_index=idx,query_norm_flag=True),full

REPEATS=47  # One model step; enough device work to hide graph-replay host overhead.

def capture(fn,count=REPEATS):
 for _ in range(3):fn()
 torch.npu.synchronize()
 graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):
  for _ in range(count):fn()
 return graph

def pair(a,b):
 out=[[],[]]
 for i in range(31):
  for j in ([0,1] if i%2==0 else [1,0]):
   start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
   start.record();(a if j==0 else b).replay();end.record();end.synchronize()
   out[j].append(start.elapsed_time(end)*1000/REPEATS)
 return [statistics.median(x) for x in out]

rows=[]
sizes=[int(x) for x in os.getenv('SIZES','1,2,7,8,16,17,31,32,33,44,63,64,65,92,127,128,129').split(',')]
for t in sizes:
 args,kw,full=make(t,True,16 if t%2 else 128)
 os.environ['MLA_PROLOG_V3_DECODE_FULL_K']='0'
 ref=C.mla_prolog_v3(*args,**kw)
 kvref=args[9].clone();krref=args[10].clone()
 packed=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=True)
 # Put canaries immediately before/after the backing allocation.
 guard=torch.full((t*20*576+64,),-19,device=dev,dtype=torch.bfloat16)
 qfull=guard[32:-32].view(t,20,576)
 packed=(qfull[...,:512],qfull[...,512:],*packed[2:])
 os.environ['MLA_PROLOG_V3_DECODE_FULL_K']='1'
 C.mla_prolog_v3_out(*args,*packed,**kw)
 torch.npu.synchronize()
 assert all(torch.equal(a,b) or a.numel()==b.numel()==0 for a,b in zip(ref,packed)), ('output',t)
 assert torch.equal(kvref,args[9]) and torch.equal(krref,args[10]),('cache',t)
 assert torch.all(guard[:32]==-19) and torch.all(guard[-32:]==-19),('canary',t)
 combined=adapter.combine_query(packed[0],packed[1])
 assert combined.data_ptr()==qfull.data_ptr() and combined.is_contiguous()
 assert torch.equal(combined,torch.cat([ref[0],ref[1]],-1))
 # Views across fewer valid tokens and BSND reshape must preserve the backing storage.
 bs=max(1,t-1)
 combined4=adapter.combine_query(packed[0][:bs].view(bs,1,20,512),packed[1][:bs].view(bs,1,20,64))
 assert combined4.data_ptr()==qfull.data_ptr() and combined4.is_contiguous()
 legacy=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True)
 def old():
  os.environ['MLA_PROLOG_V3_DECODE_FULL_K']='0'
  C.mla_prolog_v3_out(*args,*legacy,**kw)
  return torch.cat([legacy[0],legacy[1]],-1)
 def new():
  os.environ['MLA_PROLOG_V3_DECODE_FULL_K']='1'
  C.mla_prolog_v3_out(*args,*packed,**kw)
  return adapter.combine_query(packed[0],packed[1])
 ga,gb=capture(old),capture(new)
 old_us,new_us=pair(ga,gb)
 row=dict(tokens=t,bit_exact=True,canary=True,old_us=old_us,new_us=new_us,speedup=old_us/new_us)
 rows.append(row);print(json.dumps(row),flush=True)
 # Replay refreshes activations/cache slots while keeping captured output storage.
 args[0].mul_(0.5);kw['cache_index'].copy_(kw['cache_index'].flip(0))
 os.environ['MLA_PROLOG_V3_DECODE_FULL_K']='0'
 fresh=C.mla_prolog_v3(*args,**kw)
 gb.replay();torch.npu.synchronize()
 assert all(torch.equal(a,b) or a.numel()==b.numel()==0 for a,b in zip(fresh,packed)),('replay',t)

# The helper must concatenate independent tensors, even with identical shapes.
a=torch.randn(2,20,512);b=torch.randn(2,20,64)
assert torch.equal(adapter.combine_query(a,b),torch.cat([a,b],-1))
# Invalid physical strides must fail before launch.
args,kw,_=make(7)
outputs=list(allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True))
outputs[0]=torch.empty((7,20,1024),device=dev,dtype=torch.bfloat16)[...,::2]
try:C.mla_prolog_v3_out(*args,*outputs,**kw)
except RuntimeError as e:assert 'regular head strides' in str(e)
else:raise AssertionError('invalid query stride accepted')
dest=Path(os.getenv('OUT','/tmp/mla-output576-20261030/fullk-results.json'))
dest.write_text(json.dumps(dict(rows=rows,invalid_stride_rejected=True),indent=2))
print('PASS',len(rows),'decode sizes + capture replay + output guards + invalid strides',flush=True)
