"""Reproduce the actual unquantized BF16 framework input, with full precision checks.

CASE=/tmp/.../T8192.pt TOKENS=8192,32768 ASCEND_RT_VISIBLE_DEVICES=7 python3 ...
If the requested T is larger, tile complete captured sequences; this represents
multiple equal prefill sequences with the same total scheduled token count.
"""
import json,os,statistics,sys
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'python/sgl_kernel_npu'))
import sgl_kernel_npu
from sgl_kernel_npu.mla_prolog_v3_bf16_prefill import GlmMlaPrologPrefillBf16
case=torch.load(os.environ['CASE'],weights_only=True)
weights={k:v.to('npu:0') for k,v in case.items() if k not in ('token_x','rope_sin','rope_cos')}
op=GlmMlaPrologPrefillBf16(weights['weight_dq'],weights['weight_uq_qr'],weights['weight_uk'],weights['weight_dkv_kr'],weights['rmsnorm_gamma_cq'],weights['rmsnorm_gamma_ckv'])
nzweights={k:torch_npu.npu_format_cast(v,29) if k in ('weight_dq','weight_uq_qr','weight_dkv_kr') else v for k,v in weights.items()}

def bench(fn):
    for _ in range(5):fn()
    torch.npu.synchronize()
    g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(5):captured=fn()
    evs=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(21)]
    for a,b in evs:a.record();g.replay();b.record()
    torch.npu.synchronize()
    return statistics.median(a.elapsed_time(b)*1000/5 for a,b in evs)

for t in map(int,os.environ.get('TOKENS','8192,32768').split(',')):
    inputs={}
    for k in ('token_x','rope_sin','rope_cos'):
        v=case[k];inputs[k]=v.repeat((t+len(v)-1)//len(v),1)[:t].contiguous().to('npu:0')
    x,sin,cos=inputs['token_x'],inputs['rope_sin'],inputs['rope_cos']
    kv=torch.zeros((t+127)//128+1,128,1,512,device='npu:0',dtype=torch.bfloat16)
    kr=torch.zeros((t+127)//128+1,128,1,64,device='npu:0',dtype=torch.bfloat16)
    idx=torch.arange(t,device='npu:0',dtype=torch.int64)
    def reference():
        return torch.ops.npu.mla_prolog_v3(**inputs,**nzweights,kv_cache=kv,kr_cache=kr,cache_index=idx,cache_mode='PA_BSND',query_norm_flag=True,weight_quant_mode=0)
    base=reference();got=op(x,sin,cos);torch.npu.synchronize()
    refs=[base[0],base[1],base[3],kv.view(-1,512)[:t],kr.view(-1,64)[:t]]
    for name,a,b in zip(('q','qr','norm','kv','kr'),got,refs):
        a=a.float().cpu().reshape(t,-1);b=b.float().cpu().reshape(t,-1)
        dot=na=nb=err=0.;md=0.;exact=True
        for start in range(0,t,512):
            aa=a[start:start+512].double().flatten();bb=b[start:start+512].double().flatten();d=aa-bb
            dot+=torch.dot(aa,bb).item();na+=torch.dot(aa,aa).item();nb+=torch.dot(bb,bb).item();err+=torch.dot(d,d).item();md=max(md,d.abs().max().item());exact &= torch.equal(aa,bb)
        cs=dot/(na*nb)**.5 if na*nb else 1.;rel=(err/nb)**.5 if nb else 0.
        print(json.dumps(dict(tokens=t,output=name,cosine=cs,relative_l2=rel,max_abs=md,bit_exact=exact)),flush=True)
        assert cs>.99999 and rel<.005,(name,cs,rel)
    print(json.dumps(dict(tokens=t,weight_quant_mode=0,activation_dtype=str(x.dtype),fused_us=bench(reference),bf16_prepared_us=bench(lambda:op(x,sin,cos)))),flush=True)
