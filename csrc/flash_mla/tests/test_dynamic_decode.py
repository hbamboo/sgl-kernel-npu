import pytest
import torch
import torch_npu
from test_flash_mla import flash_mla,flash_mla_metadata

CONFIGS=[(40,20,5000),(48,20,5000),(56,20,5000),(64,20,5000),(2,32,3014),(4,32,3014),(5,32,3014),(6,32,3014),(7,32,3014),(10,32,3014),(15,32,3014),(1,20,3014),(3,20,3014),(8,32,1024),(16,20,3014),(17,1,256),
         (31,20,128),(32,20,1024),(33,8,128),(64,20,256),(128,20,128),
         (1,20,32768),(16,20,32768)]

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
@pytest.mark.parametrize('use_sink',[False,True])
@pytest.mark.parametrize('config',CONFIGS,ids=lambda c:f'b{c[0]}_h{c[1]}_kv{c[2]}')
def test_dynamic_lengths_after_capture(dtype,config,use_sink):
    batch,heads,maxlen=config;page=128;n=(maxlen+127)//128
    torch.manual_seed(batch+heads)
    q=torch.randn(batch,1,heads,576,device='npu',dtype=dtype)
    k=torch.randn(batch*n,page,1,576,device='npu',dtype=dtype)
    table=torch.randperm(batch*n,device='npu').to(torch.int32).reshape(batch,n)
    seq=torch.full((batch,),maxlen,dtype=torch.int32,device='npu')
    sinks=torch.randn(heads,device='npu',dtype=torch.float32) if use_sink else None
    def call():
        md=flash_mla_metadata(seq,heads,1,max_seqlen_q=1,max_seqlen_kv=maxlen)
        return flash_mla(q,k,block_table=table,cache_seqlens=seq,sinks=sinks,metadata=md,
                         softmax_scale=0.0625,max_seqlen_q=1,max_seqlen_kv=maxlen,return_softmax_lse=True)
    call();torch.npu.synchronize();graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):output,lse=call()
    for step in range(6):
        lengths=([0]*batch if step==5 else [maxlen]*batch if step==4 else
                 [min(maxlen,[0,1,15,127,128,129,255,maxlen][(i+step*3)%8]) for i in range(batch)])
        seq.copy_(torch.tensor(lengths,dtype=torch.int32,device='npu'))
        output.fill_(float('nan'));lse.fill_(float('nan'))
        graph.replay();torch.npu.synchronize()
        ref=torch.zeros_like(output,dtype=torch.float32)
        ref_lse=torch.full_like(lse,float('inf'))
        for i,length in enumerate(lengths):
            if not length:continue
            kv=k.index_select(0,table[i].long()).reshape(-1,576)[:length].float()
            scores=q[i,0].float()@kv.T*(0.0625)
            logits=torch.cat([scores,sinks[:,None]],dim=1) if use_sink else scores
            ref[i,0]=torch.softmax(logits,dim=1)[:,:length]@kv[:,:512]
            ref_lse[i,:,0]=torch.logsumexp(logits,dim=1)
        torch.testing.assert_close(output.float(),ref,atol=0.003,rtol=0.015)
        torch.testing.assert_close(lse,ref_lse,atol=0.0002,rtol=0.0002)
