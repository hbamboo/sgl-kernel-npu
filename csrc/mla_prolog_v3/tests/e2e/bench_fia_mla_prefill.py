"""Compare FIA head handling on the actual captured BF16 prefill tensors."""
import os,sys,json,statistics
from pathlib import Path
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
sys.path.insert(0,'/sgl-workspace/sglang/python')
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'python/sgl_kernel_npu'))
import torch,torch_npu
from sgl_kernel_npu.mla_prolog_v3_bf16_prefill import GlmMlaPrologPrefillBf16
from sglang.srt.hardware_backend.npu.attention.mla_fia_prefill import build_fia_mla_prefill_metadata,fia_mla_prefill
c=torch.load(os.environ['CASE'],weights_only=True);w={k:v.to('npu:0') for k,v in c.items()}
op=GlmMlaPrologPrefillBf16(w['weight_dq'],w['weight_uq_qr'],w['weight_uk'],w['weight_dkv_kr'],w['rmsnorm_gamma_cq'],w['rmsnorm_gamma_ckv'])
mask=torch.triu(torch.ones(2048,2048,dtype=torch.bool,device='npu:0'),1)
def bench(fn):
 for _ in range(3):fn()
 torch.npu.synchronize();g=torch.npu.NPUGraph()
 with torch.npu.graph(g):out=fn()
 ev=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(11)]
 for a,b in ev:a.record();g.replay();b.record()
 torch.npu.synchronize();return statistics.median(a.elapsed_time(b)*1000 for a,b in ev)
for t in map(int,os.environ.get('TOKENS','8192,32768').split(',')):
 q,qr,_,kv,kr=op(w['token_x'][:t],w['rope_sin'][:t],w['rope_cos'][:t]);cache=torch.cat((kv,kr),-1).reshape(-1,128,1,576)
 pages=t//256;table=torch.arange(t//128,dtype=torch.int32,device='npu:0').reshape(2,pages)
 meta=build_fia_mla_prefill_metadata(table,[t//2,t//2],[t//2,t//2],128)
 def call(policy):return fia_mla_prefill(q,qr,cache,meta,mask,1/16,head_policy=policy)
 a=call('pad').cpu().float().reshape(t,-1);b=call('split').cpu().float().reshape(t,-1)
 torch.set_num_threads(8);dot=na=nb=err=0.;md=0.
 for off in range(0,t,128):
  x=a[off:off+128].double().flatten();y=b[off:off+128].double().flatten();d=x-y
  dot+=torch.dot(x,y).item();na+=torch.dot(x,x).item();nb+=torch.dot(y,y).item();err+=torch.dot(d,d).item();md=max(md,d.abs().max().item())
 cs=dot/(na*nb)**.5;rel=(err/na)**.5;assert cs>.99999 and rel<.005,(cs,rel)
 print(json.dumps(dict(tokens=t,cosine=cs,relative_l2=rel,max_abs=md,pad32_us=bench(lambda:call('pad')),split16_4_us=bench(lambda:call('split')))),flush=True)
