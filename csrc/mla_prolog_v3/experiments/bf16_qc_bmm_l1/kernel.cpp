#include "mla_prolog_v3_kernel_preamble.h"
#include "kernel_operator.h"
#include "service_matmul_arch35.h"
using namespace AscendC;
using namespace MlaProlog;
#ifndef EXP_QC_N
#define EXP_QC_N 192
#endif
#ifndef EXP_QC_K
#define EXP_QC_K 64
#endif
constexpr uint32_t QN=EXP_QC_N,QK=EXP_QC_K,QKS=768/QK,STEPS=QKS*(192/QN);

#ifndef EXP_QC_K_OUTER
#define EXP_QC_K_OUTER 0
#endif
constexpr uint32_t QNS=192/QN;
__aicore__ inline uint32_t WOffset(uint32_t head,uint32_t step) {
#if EXP_QC_K_OUTER
    return head*768*192+(step/QNS)*QK*192+(step%QNS)*QN;
#else
    return head*768*192+(step%QKS)*QK*192+(step/QKS)*QN;
#endif
}
static_assert(192%QN==0 && 768%QK==0 && QN*QK<=16384,"unsupported weight tile");
#if EXP_QC_K_OUTER
static_assert(QN==64 && QK==256,"K-outer recipe validated only at N64/K256");
#else
static_assert((QN==192 && QK==64)||(QN==96 && QK==128),"unsupported N-outer recipe");
#endif

// Stage 1: [M,768] x [768,192] -> FP32 L0C.
// Fixpipe rounds to BF16 in L1; stage 2 consumes it without a GM round trip.
extern "C" __global__ __aicore__ void mla_prolog_v3_qc_bmm_l1(
    GM_ADDR norm, GM_ADDR wqc, GM_ADDR wuk, GM_ADDR output, GM_ADDR debugQc,
    uint32_t tokens, uint32_t emitQc, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if ASCEND_IS_AIC {
        TPipe pipe;
        TBuf<TPosition::A1> normBuf, weightBuf, ukBuf;
        TBuf<TPosition::C1> qcBuf;
        TBuf<TPosition::A2> a0Buf;
        TBuf<TPosition::B2> b0Buf;
        TBuf<TPosition::CO1> cBuf;
        pipe.InitBuffer(normBuf,192*1024);pipe.InitBuffer(weightBuf,64*1024);
        pipe.InitBuffer(ukBuf,192*1024);pipe.InitBuffer(qcBuf,64*1024);
        pipe.InitBuffer(a0Buf,64*1024);pipe.InitBuffer(b0Buf,64*1024);
        pipe.InitBuffer(cBuf,128*1024);
        auto a1=normBuf.Get<bfloat16_t>();auto b1=weightBuf.Get<bfloat16_t>();
        auto uk1=ukBuf.Get<bfloat16_t>();auto qc1=qcBuf.Get<bfloat16_t>();
        auto a0=a0Buf.Get<bfloat16_t>();auto b0=b0Buf.Get<bfloat16_t>();auto c0=cBuf.Get<float>();
        GlobalTensor<bfloat16_t> xg,wqg,wkg,yg,qg;
        xg.SetGlobalBuffer((__gm__ bfloat16_t*)norm);wqg.SetGlobalBuffer((__gm__ bfloat16_t*)wqc);
        wkg.SetGlobalBuffer((__gm__ bfloat16_t*)wuk);yg.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        qg.SetGlobalBuffer((__gm__ bfloat16_t*)debugQc);
        SetFlag<HardEvent::MTE1_MTE2>(3);SetFlag<HardEvent::MTE1_MTE2>(4);
        SetFlag<HardEvent::MTE1_MTE2>(5);SetFlag<HardEvent::MTE1_MTE2>(6);
        SetFlag<HardEvent::M_MTE1>(3);SetFlag<HardEvent::M_MTE1>(4);
        SetFlag<HardEvent::M_MTE1>(5);SetFlag<HardEvent::M_MTE1>(6);
        SetFlag<HardEvent::FIX_M>(3);SetFlag<HardEvent::FIX_M>(4);
        #if EXP_QC_K_OUTER
        SetFlag<HardEvent::M_MTE1>(7);
        #endif
        uint32_t tasks=((tokens+4095)/4096)*20;
        for(uint32_t task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
            uint32_t head=task%20,first=(task/20)*4096;
            uint32_t end=first+4096<tokens?first+4096:tokens;
            uint32_t firstRows=end-first<128?end-first:128;
            WaitFlag<HardEvent::MTE1_MTE2>(6);
            CopyNDGmToL1(uk1,wkg[head*192*512],192,512,512);
            SetFlag<HardEvent::MTE2_MTE1>(6);WaitFlag<HardEvent::MTE2_MTE1>(6);
            WaitFlag<HardEvent::MTE1_MTE2>(3);
            CopyNDGmToL1(a1,xg[first*768],firstRows,768,768);
            SetFlag<HardEvent::MTE2_MTE1>(3);
            for(uint32_t j=0;j<2;++j) {
                WaitFlag<HardEvent::MTE1_MTE2>(4+j);
                auto bs=b1[j*32768/2];
                CopyNDGmToL1(bs,wqg[WOffset(head,j)],QK,QN,192);
                SetFlag<HardEvent::MTE2_MTE1>(4+j);
            }
            for(uint32_t row=first;row<end;row+=128) {
                uint32_t rows=end-row<128?end-row:128,ma=(rows+15)/16*16;
                WaitFlag<HardEvent::MTE2_MTE1>(3);
                WaitFlag<HardEvent::FIX_M>(3);WaitFlag<HardEvent::FIX_M>(4);
                #if EXP_QC_K_OUTER
                for(uint32_t step=0;step<STEPS;++step) {
                    uint32_t slot=step&1,kn=step/QNS,nn=step%QNS;
                    if(nn==0) {
                        WaitFlag<HardEvent::M_MTE1>(7);
                        LoadDataL1ToL0<bfloat16_t>(a0,a1[kn*QK*ma],ma,QK);
                        SetFlag<HardEvent::MTE1_M>(7);WaitFlag<HardEvent::MTE1_M>(7);
                    }
                    WaitFlag<HardEvent::MTE2_MTE1>(4+slot);
                    WaitFlag<HardEvent::M_MTE1>(3+slot);
                    LoadDataL1ToL0<bfloat16_t,true>(b0[slot*32768/2],b1[slot*32768/2],QK,QN,QK);
                    SetFlag<HardEvent::MTE1_M>(3+slot);
                    SetFlag<HardEvent::MTE1_MTE2>(4+slot);
                    if(step+2<STEPS) {
                        WaitFlag<HardEvent::MTE1_MTE2>(4+slot);
                        auto bs=b1[slot*32768/2];
                        CopyNDGmToL1(bs,wqg[WOffset(head,step+2)],QK,QN,192);
                        SetFlag<HardEvent::MTE2_MTE1>(4+slot);
                    }
                    WaitFlag<HardEvent::MTE1_M>(3+slot);
                    MmadParams mp(ma,QN,QK,UNIT_FLAG_DISABLE,false,kn==0);
                    Mmad(c0[nn*ma*QN],a0,b0[slot*32768/2],mp);
                    SetFlag<HardEvent::M_MTE1>(3+slot);
                    if(nn+1==QNS) { SetFlag<HardEvent::M_MTE1>(7); }
                }
                SetFlag<HardEvent::M_FIX>(7);WaitFlag<HardEvent::M_FIX>(7);
                for(uint32_t nn=0;nn<QNS;++nn) {
                    FixpipeParamsArch3510<CO2Layout::NZ> local;
                    local.nSize=QN;local.mSize=ma;local.srcStride=ma;local.dstStride=ma*16;
                    local.quantPre=QuantMode_t::F322BF16;
                    Fixpipe<bfloat16_t,float,CFG_NZ>(qc1[nn*ma*QN],c0[nn*ma*QN],local);
                    if(emitQc) {
                        FixpipeParamsV220 dbg;
                        dbg.nSize=QN;dbg.mSize=rows;dbg.srcStride=ma;dbg.dstStride=3840;
                        dbg.ndNum=1;dbg.quantPre=QuantMode_t::F322BF16;
                        Fixpipe(qg[row*3840+head*192+nn*QN],c0[nn*ma*QN],dbg);
                    }
                }

                #else
                for(uint32_t ki=0;ki<STEPS;++ki) {
                    uint32_t slot=ki&1,kn=ki%QKS,nn=ki/QKS;
                    WaitFlag<HardEvent::MTE2_MTE1>(4+slot);
                    WaitFlag<HardEvent::M_MTE1>(3+slot);
                    LoadDataL1ToL0<bfloat16_t>(a0[slot*32768/2],a1[kn*QK*ma],ma,QK);
                    LoadDataL1ToL0<bfloat16_t,true>(b0[slot*32768/2],b1[slot*32768/2],QK,QN,QK);
                    SetFlag<HardEvent::MTE1_M>(3+slot);
                    SetFlag<HardEvent::MTE1_MTE2>(4+slot);
                    if(ki+2<STEPS) {
                        WaitFlag<HardEvent::MTE1_MTE2>(4+slot);
                        auto bs=b1[slot*32768/2];
                        CopyNDGmToL1(bs,wqg[WOffset(head,ki+2)],QK,QN,192);
                        SetFlag<HardEvent::MTE2_MTE1>(4+slot);
                    }
                    WaitFlag<HardEvent::MTE1_M>(3+slot);
                    MmadParams mp(ma,QN,QK,UNIT_FLAG_DISABLE,false,kn==0);
                    Mmad(c0[nn*65536/4],a0[slot*32768/2],b0[slot*32768/2],mp);
                    SetFlag<HardEvent::M_MTE1>(3+slot);
                    if(kn+1==QKS) {
                // The original prologue rounds QC to BF16 here. Preserve that
                // boundary, and emit NZ directly to the following GEMM's L1 A.
                SetFlag<HardEvent::M_FIX>(7);WaitFlag<HardEvent::M_FIX>(7);
                FixpipeParamsArch3510<CO2Layout::NZ> local;
                local.nSize=QN;local.mSize=ma;local.srcStride=ma;local.dstStride=ma*16;
                local.quantPre=QuantMode_t::F322BF16;
                Fixpipe<bfloat16_t,float,CFG_NZ>(qc1[nn*ma*QN],c0[nn*65536/4],local);
                if(emitQc) {
                    FixpipeParamsV220 dbg;
                    dbg.nSize=QN;dbg.mSize=rows;dbg.srcStride=ma;dbg.dstStride=3840;
                    dbg.ndNum=1;dbg.quantPre=QuantMode_t::F322BF16;
                    Fixpipe(qg[row*3840+head*192+nn*QN],c0[nn*65536/4],dbg);
                }
                    }
                }
                #endif
                SetFlag<HardEvent::FIX_MTE1>(7);WaitFlag<HardEvent::FIX_MTE1>(7);
                SetFlag<HardEvent::FIX_M>(3);SetFlag<HardEvent::FIX_M>(4);
                SetFlag<HardEvent::MTE1_MTE2>(3);
                // Norm and the first two projection-weight slices of the next
                // tile are prefetched while current QC is consumed by BMM.
                if(row+128<end) {
                    uint32_t nr=end-row-128<128?end-row-128:128;
                    WaitFlag<HardEvent::MTE1_MTE2>(3);
                    CopyNDGmToL1(a1,xg[(row+128)*768],nr,768,768);
                    SetFlag<HardEvent::MTE2_MTE1>(3);
                    for(uint32_t j=0;j<2;++j) {
                        WaitFlag<HardEvent::MTE1_MTE2>(4+j);
                        auto bs=b1[j*32768/2];
                        CopyNDGmToL1(bs,wqg[WOffset(head,j)],QK,QN,192);
                        SetFlag<HardEvent::MTE2_MTE1>(4+j);
                    }
                }
                #if EXP_QC_K_OUTER
                WaitFlag<HardEvent::M_MTE1>(7);
                #endif
                WaitFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(4);
                LoadDataL1ToL0<bfloat16_t>(a0,qc1,ma,192);
                SetFlag<HardEvent::MTE1_M>(3);WaitFlag<HardEvent::MTE1_M>(3);
                WaitFlag<HardEvent::M_MTE1>(5);
                LoadDataL1ToL0<bfloat16_t,true>(b0,uk1,96,128,192);
                SetFlag<HardEvent::MTE1_M>(5);
                for(uint32_t nb=0;nb<8;++nb) {
                    uint32_t slot=nb&1;
                    WaitFlag<HardEvent::MTE1_M>(5+slot);
                    if(nb+1<8) {
                        uint32_t next=slot^1;
                        WaitFlag<HardEvent::M_MTE1>(5+next);
                        LoadDataL1ToL0<bfloat16_t,true>(b0[next*32768/2],uk1[((nb+1)/2)*192*128+((nb+1)%2)*96*16],96,128,192);
                        SetFlag<HardEvent::MTE1_M>(5+next);
                    }
                    uint32_t cslot=(nb/2)&1;
                    if(nb%2==0) { WaitFlag<HardEvent::FIX_M>(3+cslot); }
                    auto c=c0[cslot*65536/4];
                    MmadParams mp(ma,128,96,nb%2==1?UNIT_FLAG_SET:UNIT_FLAG_CHECK,false,nb%2==0);
                    Mmad(c,a0[(nb%2)*ma*96],b0[slot*32768/2],mp);
                    SetFlag<HardEvent::M_MTE1>(5+slot);
                    if(nb%2==1) {
                        FixpipeParamsV220 fp;
                        fp.nSize=128;fp.mSize=rows;fp.srcStride=ma;fp.dstStride=10240;
                        fp.ndNum=1;fp.unitFlag=UNIT_FLAG_SET;fp.quantPre=QuantMode_t::F322BF16;
                        Fixpipe(yg[row*10240+head*512+(nb/2)*128],c,fp);
                        SetFlag<HardEvent::FIX_M>(3+cslot);
                    }
                }
                SetFlag<HardEvent::M_MTE1>(3);SetFlag<HardEvent::M_MTE1>(4);
                #if EXP_QC_K_OUTER
                SetFlag<HardEvent::M_MTE1>(7);
                #endif
            }
            SetFlag<HardEvent::MTE1_MTE2>(6);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(3);WaitFlag<HardEvent::MTE1_MTE2>(4);
        WaitFlag<HardEvent::MTE1_MTE2>(5);WaitFlag<HardEvent::MTE1_MTE2>(6);
        WaitFlag<HardEvent::M_MTE1>(3);WaitFlag<HardEvent::M_MTE1>(4);
        WaitFlag<HardEvent::M_MTE1>(5);WaitFlag<HardEvent::M_MTE1>(6);
        WaitFlag<HardEvent::FIX_M>(3);WaitFlag<HardEvent::FIX_M>(4);
        #if EXP_QC_K_OUTER
        WaitFlag<HardEvent::M_MTE1>(7);
        #endif
    }
}
