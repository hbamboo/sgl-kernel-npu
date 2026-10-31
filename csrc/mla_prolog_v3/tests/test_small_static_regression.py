"""Real checkpoint coverage of small decode specialization and scalar attributes."""
import os,json
from pathlib import Path
rows=[]
for layer in [0,2,23,46]:
    os.environ['REAL_WEIGHTS']='1';os.environ['LAYER']=str(layer)
    exec(Path(__file__).with_name('ablate_decode.py').read_text().split('def flags(')[0])
    assert C.mla_prolog_v3_small_decode_version>=1
    for seed in [0,42,2026]:
        torch.manual_seed(seed)
        for t in range(1,9):
            for direct_cache in [False,True]:
                for eps in [1e-6,1e-5]:
                    args,kw,full=make(t,direct_cache,16 if t%2 else 128)
                    kw.update(rmsnorm_epsilon_cq=eps,rmsnorm_epsilon_ckv=eps,
                              qc_qr_scale=1.0 if eps==1e-6 else 1.5,
                              kc_scale=1.0 if eps==1e-6 else 0.75)
                    initial=[args[9].clone(),args[10].clone()]
                    os.environ['MLA_PROLOG_V3_DECODE_STATIC']='0'
                    reference=C.mla_prolog_v3(*args,**kw)
                    caches=[args[9].clone(),args[10].clone()]
                    args[9].copy_(initial[0]);args[10].copy_(initial[1])
                    out=allocate_outputs(args[0],wdq,wuk,args[7],query_norm_flag=True,packed_query=True)
                    guard=torch.full((t*20*576+64,),-19,device=dev,dtype=torch.bfloat16)
                    qfull=guard[32:-32].view(t,20,576)
                    out=(qfull[...,:512],qfull[...,512:],*out[2:])
                    os.environ['MLA_PROLOG_V3_DECODE_STATIC']='1'
                    C.mla_prolog_v3_out(*args,*out,**kw)
                    torch.npu.synchronize()
                    case=dict(layer=layer,seed=seed,tokens=t,direct_cache=direct_cache,epsilon=eps)
                    assert all(torch.equal(x,y) or x.numel()==y.numel()==0 for x,y in zip(reference,out)),case
                    assert all(torch.equal(x,y) for x,y in zip(caches,args[9:11])),case
                    assert torch.all(guard[:32]==-19) and torch.all(guard[-32:]==-19),case
                    rows.append(dict(**case,bit_exact=True,canary=True,cache_reset_before_candidate=True))
    print('PASS real-weight small decode layer',layer,flush=True)
Path(os.environ['OUT']).write_text(json.dumps(dict(rows=rows,count=len(rows)),indent=2))
print('PASS small decode regression',len(rows),flush=True)
