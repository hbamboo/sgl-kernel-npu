import math
import os
import sys
import torch
import torch_npu
import pytest

if os.environ.get("SGL_FLASH_MLA_TEST_LIBRARY"):
    torch.ops.load_library(os.environ["SGL_FLASH_MLA_TEST_LIBRARY"])
    import importlib.util
    spec=importlib.util.spec_from_file_location("flash_mla_public_api",os.environ["SGL_FLASH_MLA_TEST_API"])
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    flash_mla,flash_mla_metadata=api.flash_mla,api.flash_mla_metadata
else:
    from sgl_kernel_npu import flash_mla,flash_mla_metadata

CASES=[
    ("BSND","PA_BBND","BSND",[1,1,1],[0,129,256],20,0,None,False),
    ("BSND","PA_BNBD","BSND",[1,1],[17,129],20,0,None,False),
    ("BSND","PA_NZ","BSND",[1,1],[127,129],20,0,None,False),
    ("BSND","PA_BBND","BSND",[3,3],[17,129],64,0,None,False),
    ("BNSD","PA_BBND","BNSD",[3,3],[17,129],20,3,None,False),
    ("BNSD","PA_BBND","BNSD",[1,1],[17,129],20,0,None,False),
    ("TND","PA_BBND","TND",[1,3,4],[17,129,256],20,3,None,False),
    ("TND","PA_BBND","NTD",[1,3,4],[17,129,256],20,3,None,False),
    ("TND","PA_BNBD","TND",[1,3,4],[17,129,256],20,4,None,False),
    ("BSND","PA_NZ","BSND",[2,2],[17,129],20,3,None,False),
    ("BSND","PA_BBND","BSND",[4,4],[17,129],20,3,[2,3],False),
    ("BSND","PA_BBND","BSND",[1,1],[17,129],64,0,None,False),
    ("TND","PA_NZ","NTD",[2,3],[17,129],96,0,None,False),
    ("BSND","PA_BBND","BSND",[1,1],[127,129],20,0,None,True),
]

def make_case(case,dtype):
    layout,kv_layout,out_layout,qlens,klens,heads,mode,used,pad=case
    torch.manual_seed(71)
    batch=len(qlens);page=128;blocks=(max(klens)+127)//128
    flat=torch.randn(sum(qlens),heads,576,device="npu",dtype=dtype)
    canonical=torch.randn(batch*blocks,page,1,576,device="npu",dtype=dtype)
    bt=torch.randperm(batch*blocks,device="npu").to(torch.int32).reshape(batch,blocks)
    if kv_layout=="PA_BBND":
        if pad:
            storage=torch.empty(batch*blocks,page+16,1,576,device="npu",dtype=dtype)
            k=storage[:,:page];k.copy_(canonical)
        else:k=canonical
    elif kv_layout=="PA_BNBD":k=canonical.permute(0,2,1,3).contiguous()
    else:k=canonical.reshape(batch*blocks,page,1,36,16).permute(0,2,3,1,4).contiguous()
    if layout=="TND":q=flat
    elif layout=="BSND":q=flat.reshape(batch,qlens[0],heads,576)
    else:q=flat.reshape(batch,qlens[0],heads,576).permute(0,2,1,3).contiguous()
    seq=torch.tensor(klens,dtype=torch.int32,device="npu")
    cu=torch.tensor([0]+list(torch.tensor(qlens).cumsum(0).tolist()),dtype=torch.int32,device="npu") if layout=="TND" else None
    used_tensor=torch.tensor(used,dtype=torch.int32,device="npu") if used is not None else None
    sink=torch.randn(heads,dtype=torch.float32,device="npu")
    mask=torch.triu(torch.ones(2048,2048,device="npu",dtype=torch.int8),1) if mode else None
    md=flash_mla_metadata(seq,heads,1,cu_seqlens_q=cu,seqused_q=used_tensor,max_seqlen_q=max(qlens),max_seqlen_kv=max(klens),mask_mode=mode,layout_q=layout,win_left=5 if mode==4 else -1,win_right=2 if mode==4 else -1)
    kwargs=dict(block_table=bt,cache_seqlens=seq,cu_seqlens_q=cu,seqused_q=used_tensor,sinks=sink,attn_mask=mask,metadata=md,
                softmax_scale=576**-0.5,mask_mode=mode,win_left=5 if mode==4 else -1,win_right=2 if mode==4 else -1,
                max_seqlen_q=max(qlens),max_seqlen_kv=max(klens),layout_q=layout,layout_kv=kv_layout,layout_out=out_layout,return_softmax_lse=True)
    return q,k,kwargs,canonical,flat

def golden(case,canonical,flat,bt,sinks):
    layout,kv_layout,out_layout,qlens,klens,heads,mode,used,pad=case
    kv=canonical.cpu().float();query=flat.cpu().float();table=bt.cpu();sink=sinks.cpu()
    out=torch.zeros(sum(qlens),heads,512,dtype=torch.float32)
    lse=torch.full((heads,sum(qlens)),float("inf"));offset=0
    for b,(physical,length) in enumerate(zip(qlens,klens)):
        active=used[b] if used else physical
        if length>0 and active>0:
            key=kv.index_select(0,table[b].long()).reshape(-1,576)[:length]
            scores=query[offset:offset+active].permute(1,0,2)@key.T*(576**-0.5)
            pos=torch.arange(active)[:,None]+length-active;kp=torch.arange(length)[None,:]
            if mode==3:scores.masked_fill_((kp>pos)[None,:,:],float('-inf'))
            elif mode==4:scores.masked_fill_(((kp<pos-5)|(kp>pos+2))[None,:,:],float('-inf'))
            logits=torch.cat([scores,sink[:,None,None].expand(heads,active,1)],dim=-1)
            probs=torch.softmax(logits,-1)[...,:length]
            out[offset:offset+active]=(probs@key[:,:512]).permute(1,0,2)
            lse[:,offset:offset+active]=torch.logsumexp(logits,-1)
        offset+=physical
    if layout!="TND":
        out=out.reshape(len(qlens),qlens[0],heads,512)
        lse=lse.reshape(heads,len(qlens),qlens[0]).permute(1,0,2)
        if layout=="BNSD":out=out.permute(0,2,1,3)
    elif out_layout=="NTD":out=out.permute(1,0,2)
    return out,lse

@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
@pytest.mark.parametrize("case",CASES)
def test_layout_precision(case,dtype):
    q,k,kw,canonical,flat=make_case(case,dtype)
    output,lse=flash_mla(q,k,**kw)
    expected,expected_lse=golden(case,canonical,flat,kw['block_table'],kw['sinks'])
    torch.testing.assert_close(output.cpu().float(),expected,atol=0.003,rtol=0.015)
    torch.testing.assert_close(lse.cpu(),expected_lse,atol=0.0002,rtol=0.0002)

def test_multistream_graph():
    case=CASES[0];items=[]
    for _ in range(2):
        q,k,kw,canonical,flat=make_case(case,torch.bfloat16)
        stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            flash_mla(q,k,**kw)
        stream.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream):output,lse=flash_mla(q,k,**kw)
        items.append((stream,g,q,k,kw,canonical,flat,output,lse))
    for _ in range(3):
        for stream,g,*_rest in items:
            with torch.npu.stream(stream):g.replay()
    torch.npu.synchronize()
    for stream,g,q,k,kw,canonical,flat,output,lse in items:
        expected,expected_lse=golden(case,canonical,flat,kw['block_table'],kw['sinks'])
        torch.testing.assert_close(output.cpu().float(),expected,atol=0.003,rtol=0.015)
        torch.testing.assert_close(lse.cpu(),expected_lse,atol=0.0002,rtol=0.0002)

def test_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        q=torch.empty(7,20,576,device="npu",dtype=torch.bfloat16);k=torch.empty(4,128,1,576,device="npu",dtype=q.dtype)
        o,l=flash_mla(q,k,layout_q="TND",layout_out="NTD",return_softmax_lse=False)
        assert o.shape==(20,7,512) and l.shape==(0,)
        o,l=flash_mla(q,k,layout_q="TND",layout_out="TND",return_softmax_lse=True)
        assert o.shape==(7,20,512) and l.shape==(20,7)
        seq=torch.empty(2,device="npu",dtype=torch.int32)
        assert flash_mla_metadata(seq,20,1,max_seqlen_q=1).numel()==4096

def test_no_legacy_python_import():
    assert "cann_ops_transformer" not in sys.modules

def test_default_limits_and_empty_query():
    case=CASES[0]
    q,k,kw,canonical,flat=make_case(case,torch.bfloat16)
    kw.pop('max_seqlen_q');kw.pop('max_seqlen_kv');kw['metadata']=None
    output,lse=flash_mla(q,k,**kw)
    expected,expected_lse=golden(case,canonical,flat,kw['block_table'],kw['sinks'])
    torch.testing.assert_close(output.cpu().float(),expected,atol=0.003,rtol=0.015)
    torch.testing.assert_close(lse.cpu(),expected_lse,atol=0.0002,rtol=0.0002)
    output,lse=flash_mla(q[:,:0],k,**kw)
    assert output.shape==(3,0,20,512) and lse.shape==(3,20,0)
