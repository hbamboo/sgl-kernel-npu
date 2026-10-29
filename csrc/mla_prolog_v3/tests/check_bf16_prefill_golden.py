"""Actual captured BF16 input vs staged FP64 CPU golden, sampled rows plus sequence boundary.
CASE must name a captured framework .pt file; visible device is chosen by caller.
"""
import torch,torch_npu,sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'python/sgl_kernel_npu'))
import sgl_kernel_npu
from sgl_kernel_npu.mla_prolog_v3_bf16_prefill import GlmMlaPrologPrefillBf16
c=torch.load(os.environ['CASE'],weights_only=True)
w={k:v.to('npu:0') for k,v in c.items()};t=len(c['token_x'])
impl=GlmMlaPrologPrefillBf16(w['weight_dq'],w['weight_uq_qr'],w['weight_uk'],w['weight_dkv_kr'],w['rmsnorm_gamma_cq'],w['rmsnorm_gamma_ckv'])
kv=torch.empty(((t+127)//128,128,1,512),device='npu:0',dtype=torch.bfloat16);kr=torch.empty(((t+127)//128,128,1,64),device='npu:0',dtype=torch.bfloat16)
a={k:torch_npu.npu_format_cast(v,29) if k in ('weight_dq','weight_uq_qr','weight_dkv_kr') else v for k,v in w.items()}
base=torch.ops.npu.mla_prolog_v3(**a,kv_cache=kv,kr_cache=kr,cache_index=torch.arange(t,device='npu:0'),query_norm_flag=True,weight_quant_mode=0,cache_mode='PA_BSND')
got=impl(w['token_x'],w['rope_sin'],w['rope_cos'])
rows=torch.cat((torch.linspace(0,t-1,min(t,128)).long(),torch.tensor([max(t//2-1,0)]))).unique();r=rows.to('npu:0')
ref=[base[0][r].cpu(),base[1][r].cpu(),base[3][r].cpu(),kv.view(-1,512)[r].cpu(),kr.view(-1,64)[r].cpu()]
new=[v[r].cpu() for v in got]
torch.set_num_threads(8)
x=c['token_x'][rows].double()
def norm(v,g):
    v=v.double();return (v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)*g.double()).bfloat16()
cq=(x@c['weight_dq'].double()).bfloat16();qn=norm(cq,c['rmsnorm_gamma_cq'])
uq=(qn.double()@c['weight_uq_qr'].double()).bfloat16().view(-1,20,256)
q=torch.einsum('tnd,ndh->tnh',uq[:,:,:192].double(),c['weight_uk'].double()).bfloat16()
ck=(x@c['weight_dkv_kr'].double()).bfloat16()
s=c['rope_sin'][rows].double();co=c['rope_cos'][rows].double()
def rope(v):
    a=v[...,::2].double();b=v[...,1::2].double();v=torch.cat((a,b),-1);rotate=torch.cat((-b,a),-1)
    shape=(len(rows),)+(1,)*(v.dim()-2)+(64,)
    return (v*co.view(shape)+rotate*s.view(shape)).bfloat16()
gold=[q,rope(uq[:,:,192:]),qn,norm(ck[:,:512],c['rmsnorm_gamma_ckv']),rope(ck[:,512:])]
for name,b,n,g in zip(('q','qr','norm','kv','kr'),ref,new,gold):
    for source,v in [('raw',b),('native',n)]:
        vv=v.double().flatten();gg=g.double().flatten();err=torch.linalg.vector_norm(vv-gg)/torch.linalg.vector_norm(gg);cos=torch.dot(vv,gg)/(torch.linalg.vector_norm(vv)*torch.linalg.vector_norm(gg))
        print(json.dumps(dict(output=name,source=source,cosine=cos.item(),relative_l2=err.item(),max_abs=(vv-gg).abs().max().item())),flush=True)
        assert cos>.99999 and err<.005,(name,source,cos,err)
