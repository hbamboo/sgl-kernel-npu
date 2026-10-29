"""FIA absorbed MLA: compact pages, causal prefixes, padded heads/tokens, routing."""
import os,math,sys,json
from types import SimpleNamespace as NS
from unittest.mock import patch
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7','use physical NPU 7'
sys.path.insert(0,'/sgl-workspace/sglang/python')
import torch,torch_npu
from sglang.srt.hardware_backend.npu.attention.mla_fia_prefill import build_fia_mla_prefill_metadata,fia_mla_prefill
from sglang.srt.models.deepseek_common import attention_backend_handler as handler
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import AttnForwardMethod

torch.set_num_threads(8)
def route_test():
    def mode(extend):return NS(is_extend=lambda:extend,is_target_verify=lambda:False,is_draft_extend_v2=lambda:False)
    for extend,flash,independent,dsa,want in [(True,True,True,False,AttnForwardMethod.MLA_NPU),(True,False,True,False,AttnForwardMethod.MLA_NPU),(True,False,False,False,AttnForwardMethod.MHA_NPU),(False,False,True,False,AttnForwardMethod.MLA_NPU),(True,False,True,True,AttnForwardMethod.DSA_NPU)]:
        backend=NS(use_flash_mla_prefill=flash,use_mlaprolog_v3_prefill=independent)
        with patch.object(handler,'get_attn_backend',return_value=backend),patch.object(handler.envs.ENABLE_ASCEND_FLASH_MLA_PREFILL,'get',return_value=flash):
            assert handler.handle_attention_ascend(NS(use_dsa=dsa),NS(forward_mode=mode(extend)))==want
    print('routing: 5 passed',flush=True)


def run(ql,kl,heads=20,pad=3):
    torch.manual_seed(77+sum(ql));page=128;counts=[(x+page-1)//page for x in kl]
    blocks=sum(counts)+3;cache=torch.randn(blocks,page,1,576,dtype=torch.bfloat16)*.2
    table=torch.full((len(ql),max(counts)),blocks+99,dtype=torch.int32)
    order=torch.randperm(blocks-1)+1;off=0
    for i,n in enumerate(counts):
        table[i,:n]=order[off:off+n];off+=n
    q=torch.randn(sum(ql)+pad,heads,512,dtype=torch.bfloat16)*.2
    qr=torch.randn(sum(ql)+pad,heads,64,dtype=torch.bfloat16)*.2
    mask=torch.triu(torch.ones(2048,2048,dtype=torch.bool),1).to('npu:0')
    meta=build_fia_mla_prefill_metadata(table.to('npu:0'),ql,kl,page)
    out=fia_mla_prefill(q.to('npu:0'),qr.to('npu:0'),torch.stack((torch.zeros_like(cache),cache)).to('npu:0')[1],meta,mask,1/16).view(-1,heads,512).float().cpu()
    assert meta.page_indices.numel()==sum(n for n,nq in zip(counts,ql) if nq)
    assert torch.count_nonzero(out[sum(ql):])==0
    gold=[];oq=0
    for i,(nq,nk) in enumerate(zip(ql,kl)):
        if not nq:continue
        kv=cache[table[i,:counts[i]].long()].reshape(-1,576)[:nk].float()
        k,kr=kv[:,:512],kv[:,512:]
        score=(torch.einsum('qnd,kd->nqk',q[oq:oq+nq].float(),k)+torch.einsum('qnd,kd->nqk',qr[oq:oq+nq].float(),kr))/16
        causal=torch.arange(nk).view(1,-1)>(nk-nq+torch.arange(nq).view(-1,1));score.masked_fill_(causal.unsqueeze(0),float('-inf'))
        gold.append(torch.einsum('nqk,kd->qnd',score.softmax(-1),k));oq+=nq
    a=out[:sum(ql)].double().flatten();b=torch.cat(gold).double().flatten();cs=torch.dot(a,b)/(a.norm()*b.norm());rel=(a-b).norm()/b.norm()
    print(json.dumps(dict(query_lens=ql,kv_lens=kl,heads=heads,cosine=cs.item(),relative_l2=rel.item(),max_abs=(a-b).abs().max().item())),flush=True)
    assert cs>.99999 and rel<.005

def invalid_metadata_tests():
    table=torch.tensor([[2,999]],dtype=torch.int32)
    try:build_fia_mla_prefill_metadata(table,[5],[4],128)
    except ValueError:pass
    else:raise AssertionError('query > KV was accepted')
    z=build_fia_mla_prefill_metadata(table,[0],[100],128)
    assert z.num_tokens==0 and z.page_indices.numel()==0

invalid_metadata_tests()
route_test()
for q,k,h in [([5,7],[5,7],20),([3,2],[5,4],20),([33,17],[145,257],20),([257,129],[320,200],20),([1,0,9],[129,511,260],20),([3],[129],32),([3],[7],1)]:run(q,k,h)
print('ALL PASSED',flush=True)
