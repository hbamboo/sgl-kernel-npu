"""Measure all five outputs of the full unquantized GLM prefill path."""
import os,json,torch,torch_npu
import check as h

def fused_prefill(x,sin,cos):
    op=h.op;t=len(x);co=cos.view(t,1,1,64);si=sin.view(t,1,1,64)
    main=torch.npu.current_stream(x.device)
    evcq,evnorm,evkv,evqr,evdone=op.events
    cq=torch.mm(x,op.w_dq);evcq.record(main)
    with torch.npu.stream(op.aux):
        op.aux.wait_event(evcq)
        norm=torch_npu.npu_rms_norm(cq,op.gq,epsilon=op.eq)[0]
        evnorm.record(op.aux)
    ckvkr=torch.mm(x,op.w_kv);evkv.record(main)
    with torch.npu.stream(op.aux):
        op.aux.wait_event(evkv)
        kv=torch_npu.npu_rms_norm(ckvkr[:,:512],op.gkv,epsilon=op.ekv)[0]
        kr=torch_npu.npu_rotary_mul(ckvkr[:,512:].reshape(t,1,1,64),co,si).view(t,64)
    main.wait_event(evnorm)
    # QR is independent of QC; launch its RoPE before the fused Cube stages.
    qr=torch.mm(norm,op.w_qr).view(t,20,1,64);evqr.record(main)
    with torch.npu.stream(op.aux):
        op.aux.wait_event(evqr)
        qrope=torch_npu.npu_rotary_mul(qr,co,si).view(t,20,64)
        evdone.record(op.aux)
    q,_=torch.ops.npu.mla_prolog_v3_qc_bmm_l1(norm,h.packed,op.w_uk,False)
    main.wait_event(evdone)
    for value in (cq,ckvkr,qr,si,co):value.record_stream(op.aux)
    for value in (norm,qrope,kv,kr):value.record_stream(main)
    return tuple(value.contiguous() for value in (q,qrope,norm,kv,kr))

if __name__=='__main__':
    for t in map(int,os.environ.get('TOKENS','8192,32768').split(',')):
        x=h.w['token_x'][:t].contiguous();sin=h.w['rope_sin'][:t].contiguous();cos=h.w['rope_cos'][:t].contiguous()
        a=h.op(x,sin,cos);b=fused_prefill(x,sin,cos);torch.npu.synchronize()
        print('full_pipeline_tokens',t,flush=True)
        for name,u,v in zip(('q','qr','norm','kv','kr'),b,a):h.stats(name,u,v)
        print(json.dumps(dict(tokens=t,native_pipeline_us=h.bench(lambda:h.op(x,sin,cos)),fused_l1_pipeline_us=h.bench(lambda:fused_prefill(x,sin,cos)))),flush=True)
