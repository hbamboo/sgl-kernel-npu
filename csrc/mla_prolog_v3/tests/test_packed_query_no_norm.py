"""The optional query_norm output must not change packed Query values."""
import os
from pathlib import Path
exec(Path(__file__).with_name('test_packed_query.py').read_text().split('rows=[]')[0])
for t in [1,17,64,92,129]:
    args,kw,_=make(t,True)
    kw['query_norm_flag']=False
    os.environ['MLA_PROLOG_V3_DECODE_TUNE']='0'
    ref=C.mla_prolog_v3(*args,**kw)
    os.environ['MLA_PROLOG_V3_DECODE_TUNE']='1'
    out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=False,packed_query=True)
    C.mla_prolog_v3_out(*args,*out,**kw)
    torch.npu.synchronize()
    assert all(torch.equal(a,b) or a.numel()==b.numel()==0 for a,b in zip(ref,out)),t
    assert out[3].numel()==out[4].numel()==0
    print('PASS query_norm_flag=False',t,flush=True)
