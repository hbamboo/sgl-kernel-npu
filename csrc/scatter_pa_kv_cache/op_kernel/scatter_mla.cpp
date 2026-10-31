// Fused MLA K-nope / V split, K-rope broadcast and paged K/V write.
#include "kernel_operator.h"
#include "scatter_mla_tiling.h"

using namespace AscendC;
namespace sglang::scatter_mla {
__aicore__ inline int64_t ScalarMin(int64_t a,int64_t b) {return a<b?a:b;}
__simd_vf__ inline void AssembleRope(__ubuf__ uint16_t* key,__ubuf__ uint16_t* rope,
                                    uint32_t tokens,uint32_t heads,uint32_t dim,
                                    uint32_t nopeDim,uint32_t ropeDim) {
    for(uint16_t t=0;t<tokens;++t) {
        for(uint16_t c=0;c<ropeDim;c+=128) {
            MicroAPI::RegTensor<uint16_t> value;
            uint32_t remaining=ropeDim-c;
            auto mask=MicroAPI::UpdateMask<uint16_t>(remaining);
            MicroAPI::DataCopy(value,rope+t*ropeDim+c);
            for(uint16_t h=0;h<heads;++h) {
                MicroAPI::DataCopy(key+(t*heads+h)*dim+nopeDim+c,value,mask);
            }
        }
    }
}

template<class Index>
class Kernel {
    GlobalTensor<uint16_t> packedGm_,ropeGm_,keyCacheGm_,valueCacheGm_;
    GlobalTensor<uint16_t> keyOutGm_,valueOutGm_;
    GlobalTensor<Index> slotsGm_;
    TQueBind<QuePosition::VECIN,QuePosition::VECOUT,2> keyQ_,valueQ_;
    TQue<QuePosition::VECIN,2> ropeQ_;
    TBuf<TPosition::VECCALC> slotsBuf_;
    TPipe pipe_;
    ScatterMlaTiling p_;
    int64_t start_,count_,dim_,packedDim_;
public:
    __aicore__ inline void Init(GM_ADDR packed,GM_ADDR rope,GM_ADDR slots,GM_ADDR kc,GM_ADDR vc,
                               GM_ADDR keyOut,GM_ADDR valueOut,const ScatterMlaTiling& p) {
        p_=p;start_=GetBlockIdx()*p_.tokensPerCore;
        count_=ScalarMin(p_.tokensPerCore,p_.tokens-start_);
        dim_=p_.nopeDim+p_.ropeDim;packedDim_=p_.nopeDim+p_.valueDim;
        packedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(packed));
        ropeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(rope));
        slotsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Index*>(slots));
        keyCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(kc));
        valueCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(vc));
        if(p_.writeDense) {
            keyOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keyOut));
            valueOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(valueOut));
        }
        pipe_.InitBuffer(keyQ_,2,p_.tileTokens*p_.heads*dim_*sizeof(uint16_t));
        pipe_.InitBuffer(valueQ_,2,p_.tileTokens*p_.heads*p_.valueDim*sizeof(uint16_t));
        // Extra vector register protects the final masked 64-lane rope load.
        pipe_.InitBuffer(ropeQ_,2,p_.tileTokens*p_.ropeDim*sizeof(uint16_t)+256);
        pipe_.InitBuffer(slotsBuf_,((p_.tokensPerCore*sizeof(Index)+31)/32)*32);
    }
    __aicore__ inline void CopyIn(int64_t off,int64_t n) {
        auto k=keyQ_.template AllocTensor<uint16_t>();
        auto v=valueQ_.template AllocTensor<uint16_t>();
        auto rope=ropeQ_.template AllocTensor<uint16_t>();
        DataCopyPadExtParams<uint16_t> pad{false,0,0,0};
        DataCopyExtParams kp{static_cast<uint16_t>(n*p_.heads),static_cast<uint32_t>(p_.nopeDim*2),
                             static_cast<uint32_t>(p_.valueDim*2),static_cast<uint32_t>(p_.ropeDim*2/32),0};
        DataCopyExtParams vp{static_cast<uint16_t>(n*p_.heads),static_cast<uint32_t>(p_.valueDim*2),
                             static_cast<uint32_t>(p_.nopeDim*2),0,0};
        DataCopyExtParams rp{1,static_cast<uint32_t>(n*p_.ropeDim*2),0,0,0};
        DataCopyPad(k,packedGm_[(start_+off)*p_.heads*packedDim_],kp,pad);
        DataCopyPad(v,packedGm_[(start_+off)*p_.heads*packedDim_+p_.nopeDim],vp,pad);
        DataCopyPad(rope,ropeGm_[(start_+off)*p_.ropeDim],rp,pad);
        keyQ_.EnQue(k);valueQ_.EnQue(v);ropeQ_.EnQue(rope);
    }
    __aicore__ inline void ComputeOut(int64_t off,int64_t n,const LocalTensor<Index>& slots) {
        auto rope=ropeQ_.template DeQue<uint16_t>();
        auto k=keyQ_.template DeQue<uint16_t>();
        auto v=valueQ_.template DeQue<uint16_t>();
        asc_vf_call<AssembleRope>(reinterpret_cast<__ubuf__ uint16_t*>(k.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ uint16_t*>(rope.GetPhyAddr()),
                                 static_cast<uint32_t>(n),static_cast<uint32_t>(p_.heads),
                                 static_cast<uint32_t>(dim_),static_cast<uint32_t>(p_.nopeDim),
                                 static_cast<uint32_t>(p_.ropeDim));
        auto event=static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);WaitFlag<HardEvent::V_MTE3>(event);
        // Reuse the assembled UB rows for contiguous FIA inputs and PA caches.
        // Dense rows are produced even when a padding slot is invalid for PA.
        if(p_.writeDense) {
            DataCopyExtParams kp{1,static_cast<uint32_t>(n*p_.heads*dim_*2),0,0,0};
            DataCopyExtParams vp{1,static_cast<uint32_t>(n*p_.heads*p_.valueDim*2),0,0,0};
            DataCopyPad(keyOutGm_[(start_+off)*p_.heads*dim_],k,kp);
            DataCopyPad(valueOutGm_[(start_+off)*p_.heads*p_.valueDim],v,vp);
        }
        for(int64_t i=0;i<n;) {
            int64_t slot=slots.GetValue(off+i);
            if(slot<0||slot>=p_.capacity) {++i;continue;}
            int64_t run=1;
            if(p_.coalesce) {
                while(i+run<n&&slot+run<p_.capacity&&slots.GetValue(off+i+run)==slot+run)++run;
            }
            DataCopyExtParams kp{1,static_cast<uint32_t>(run*p_.heads*dim_*2),0,0,0};
            DataCopyExtParams vp{1,static_cast<uint32_t>(run*p_.heads*p_.valueDim*2),0,0,0};
            DataCopyPad(keyCacheGm_[slot*p_.heads*dim_],k[i*p_.heads*dim_],kp);
            DataCopyPad(valueCacheGm_[slot*p_.heads*p_.valueDim],v[i*p_.heads*p_.valueDim],vp);
            i+=run;
        }
        keyQ_.FreeTensor(k);valueQ_.FreeTensor(v);ropeQ_.FreeTensor(rope);
    }
    __aicore__ inline void Process() {
        if(GetBlockIdx()>=p_.usedCores||count_<=0)return;
        auto slots=slotsBuf_.template Get<Index>();
        DataCopyExtParams sp{1,static_cast<uint32_t>(count_*sizeof(Index)),0,0,0};
        DataCopyPadExtParams<Index> pad{false,0,0,0};
        DataCopyPad(slots,slotsGm_[start_],sp,pad);
        auto e=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(e);WaitFlag<HardEvent::MTE2_S>(e);
        CopyIn(0,ScalarMin(p_.tileTokens,count_));
        for(int64_t off=0;off<count_;off+=p_.tileTokens) {
            int64_t next=off+p_.tileTokens;
            if(next<count_)CopyIn(next,ScalarMin(p_.tileTokens,count_-next));
            ComputeOut(off,ScalarMin(p_.tileTokens,count_-off),slots);
        }
    }
};
}
extern "C" __global__ __aicore__ void sgl_scatter_mla(GM_ADDR packed,GM_ADDR rope,GM_ADDR slots,
                                                     GM_ADDR kc,GM_ADDR vc,GM_ADDR keyOut,
                                                     GM_ADDR valueOut,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScatterMlaTiling p;
    auto* dst=reinterpret_cast<uint64_t*>(&p);
    auto* src=reinterpret_cast<__gm__ uint64_t*>(tiling);
    for(int i=0;i<sizeof(p)/sizeof(uint64_t);++i)dst[i]=src[i];
    if(p.indexBytes==4) {
        sglang::scatter_mla::Kernel<int32_t> op;op.Init(packed,rope,slots,kc,vc,keyOut,valueOut,p);op.Process();
    } else {
        sglang::scatter_mla::Kernel<int64_t> op;op.Init(packed,rope,slots,kc,vc,keyOut,valueOut,p);op.Process();
    }
}
