"""Unquantized GLM prefill: BF16 activation, weights and intermediates.

This path introduces no activation or weight quantization. Prepare the four
BF16 weights once, then overlap independent matrix/vector stages on two streams.
One instance belongs to one serialized caller (e.g. one attention layer).
The BF16 rounding order can differ from the fused kernel; compare with the
staged golden and application quality before enabling this experimental path.
"""
from contextlib import nullcontext
import os
import threading
import torch
import torch_npu


_STREAMS = {}
_STREAM_LOCK = threading.Lock()

def _aux_stream(device):
    # Reuse one auxiliary stream across layers; otherwise each stream retains
    # its own large allocator cache at a 32768-token prefill.
    main = torch.npu.current_stream(device)
    key = (device.index, threading.get_ident(), main.npu_stream)
    with _STREAM_LOCK:
        if key not in _STREAMS:
            _STREAMS[key] = torch.npu.Stream(device=device)
        return _STREAMS[key]


def _weight(w,k,n):
    if w.dtype != torch.bfloat16:
        raise ValueError('unquantized prefill requires BF16 weights')
    if torch_npu.get_npu_format(w) not in (0,2):
        w=torch_npu.npu_format_cast(w,2)
    if tuple(w.shape)==(n,k):w=w.t()
    if tuple(w.shape)!=(k,n):raise ValueError('GLM BF16 weight shape mismatch')
    return w.t().contiguous().t()


class GlmMlaPrologPrefillBf16:
    def __init__(self,w_dq,w_uq_qr,w_uk,w_dkv_kr,gamma_cq,gamma_ckv,
                 epsilon_cq=1e-5,epsilon_ckv=1e-5,parallel=True):
        self.w_dq=_weight(w_dq,2048,768)
        uq=_weight(w_uq_qr,768,5120).reshape(768,20,256)
        self.w_qc=_weight(uq[:,:,:192].contiguous().view(768,3840),768,3840)
        qr=uq[:,:,192:]
        self.w_qr=_weight(torch.cat((qr[:,:,::2],qr[:,:,1::2]),2).contiguous().view(768,1280),768,1280)
        kv=_weight(w_dkv_kr,2048,576)
        kr=kv[:,512:]
        self.w_kv=_weight(torch.cat((kv[:,:512],kr[:,::2],kr[:,1::2]),1),2048,576)
        if tuple(w_uk.shape)!=(20,192,512) or w_uk.dtype!=torch.bfloat16:
            raise ValueError('GLM BF16 Wuk must be [20,192,512]')
        self.w_uk=w_uk.contiguous()
        for gamma, width in ((gamma_cq,768),(gamma_ckv,512)):
            if gamma.dtype != torch.bfloat16 or tuple(gamma.shape) != (width,):
                raise ValueError("GLM BF16 normalization weight shape/dtype mismatch")
        self.gq,self.gkv=gamma_cq,gamma_ckv
        self.eq,self.ekv=epsilon_cq,epsilon_ckv
        self.parallel=parallel
        self.aux=_aux_stream(w_uk.device) if parallel else None
        self.events=[torch.npu.Event() for _ in range(5)] if parallel else None

    def __call__(self,x,sin,cos):
        if x.dtype!=torch.bfloat16 or x.dim()!=2 or x.shape[1]!=2048:
            raise ValueError('unquantized GLM prefill expects BF16 [T,2048]')
        t=x.shape[0]
        if tuple(sin.shape)!=(t,64) or tuple(cos.shape)!=(t,64):
            raise ValueError('RoPE table must be [T,64]')
        cos=cos.view(t,1,1,64);sin=sin.view(t,1,1,64)
        main=torch.npu.current_stream(x.device)
        if self.parallel:evcq,evnorm,evkv,evqr,evdone=self.events
        cq=torch.mm(x,self.w_dq)
        if self.parallel:evcq.record(main)
        with torch.npu.stream(self.aux) if self.parallel else nullcontext():
            if self.parallel:self.aux.wait_event(evcq)
            norm=torch_npu.npu_rms_norm(cq,self.gq,epsilon=self.eq)[0]
            if self.parallel:evnorm.record(self.aux)
        ckvkr=torch.mm(x,self.w_kv)
        if self.parallel:evkv.record(main)
        with torch.npu.stream(self.aux) if self.parallel else nullcontext():
            if self.parallel:self.aux.wait_event(evkv)
            kv=torch_npu.npu_rms_norm(ckvkr[:,:512],self.gkv,epsilon=self.ekv)[0]
            kr=torch_npu.npu_rotary_mul(ckvkr[:,512:].reshape(t,1,1,64),cos,sin).view(t,64)
        if self.parallel:main.wait_event(evnorm)
        qc=torch.mm(norm,self.w_qc).view(t,20,192)
        qr=torch.mm(norm,self.w_qr).view(t,20,1,64)
        if self.parallel:evqr.record(main)
        with torch.npu.stream(self.aux) if self.parallel else nullcontext():
            if self.parallel:self.aux.wait_event(evqr)
            qrope=torch_npu.npu_rotary_mul(qr,cos,sin).view(t,20,64)
            if self.parallel:evdone.record(self.aux)
        q=torch_npu.npu_transpose_batchmatmul(qc,self.w_uk,
            perm_x1=(1,0,2),perm_x2=(0,1,2),perm_y=(1,0,2))
        if self.parallel:
            main.wait_event(evdone)
            for value in (cq,ckvkr,qr,sin,cos):
                value.record_stream(self.aux)
            for value in (norm,qrope,kv,kr):
                value.record_stream(main)
        result=tuple(value.contiguous() for value in (q,qrope,norm,kv,kr))
        if os.environ.get("SGLANG_NPU_MLAPO_V3_BF16_DIAG_SYNC") == "1":
            torch.npu.synchronize(x.device)
        return result
