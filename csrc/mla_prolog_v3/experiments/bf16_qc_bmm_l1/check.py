"""Actual BF16 CQ projection + BMM, with the intermediate rounding inspected.

ASCEND_RT_VISIBLE_DEVICES=7 LIB=/tmp/bk-qc-bmm-l1/libfused_qc_bmm.so CASE=...pt python3 check.py
"""
import os,json,statistics,sys
from pathlib import Path
if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "7":
    raise RuntimeError("This validation must run with ASCEND_RT_VISIBLE_DEVICES=7")
import torch,torch_npu
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'python/sgl_kernel_npu'))
from sgl_kernel_npu.mla_prolog_v3_bf16_prefill import GlmMlaPrologPrefillBf16
lib=os.environ.get('LIB','/tmp/bk-qc-bmm-l1/libfused_qc_bmm.so')
torch.ops.load_library(lib)
case=torch.load(os.environ['CASE'],weights_only=True)
w={k:v.to('npu:0') for k,v in case.items()}
op=GlmMlaPrologPrefillBf16(w['weight_dq'],w['weight_uq_qr'],w['weight_uk'],w['weight_dkv_kr'],w['rmsnorm_gamma_cq'],w['rmsnorm_gamma_ckv'])
packed=op.w_qc.reshape(768,20,192).permute(1,0,2).contiguous()
with torch.no_grad():
    norm=torch_npu.npu_rms_norm(torch.mm(w['token_x'],op.w_dq),op.gq,epsilon=op.eq)[0]

def stats(name,a,b):
    a=a.cpu().float().reshape(len(a),-1);b=b.cpu().float().reshape(len(b),-1)
    dot=na=nb=err=0.;md=0.
    for off in range(0,len(a),128):
        x=a[off:off+128].double().flatten();y=b[off:off+128].double().flatten();d=x-y
        dot+=torch.dot(x,y).item();na+=torch.dot(x,x).item();nb+=torch.dot(y,y).item();err+=torch.dot(d,d).item();md=max(md,d.abs().max().item())
    cs=dot/(na*nb)**.5 if na*nb else 1.;rel=(err/nb)**.5 if nb else 0.
    print(json.dumps(dict(output=name,cosine=cs,relative_l2=rel,max_abs=md,bit_exact=torch.equal(a,b))),flush=True)
    assert cs>.99999 and rel<.005,(name,cs,rel)
    if os.environ.get("STRICT")=="1":assert torch.equal(a,b),name

def bench(fn):
    for _ in range(5):fn()
    torch.npu.synchronize();g=torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for _ in range(5):out=fn()
    es=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(21)]
    for a,b in es:a.record();g.replay();b.record()
    torch.npu.synchronize()
    return statistics.median(a.elapsed_time(b)*1000/5 for a,b in es)

if __name__ == "__main__":
    for t in map(int,os.environ.get('TOKENS','1,7,92,128,129,8192,32768').split(',')):
        n=norm[:t].contiguous();assert len(n)==t
        if os.environ.get('RANDOM')=='1':
            torch.manual_seed(3+t);n=torch.randn_like(n)*.5
        def native():
            q=torch.mm(n,op.w_qc).view(t,20,192)
            return torch_npu.npu_transpose_batchmatmul(q,op.w_uk,perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2)),q
        print('tokens',t,flush=True)
        y,qc=torch.ops.npu.mla_prolog_v3_qc_bmm_l1(n,packed,op.w_uk,True)
        refq,refqc=native();torch.npu.synchronize()
        stats('qc',qc,refqc);stats('q',y,refq)
        if t>=8192:
            a=bench(native);b=bench(lambda:torch.ops.npu.mla_prolog_v3_qc_bmm_l1(n,packed,op.w_uk,False))
            print(json.dumps(dict(tokens=t,native_us=a,fused_l1_us=b)),flush=True)
