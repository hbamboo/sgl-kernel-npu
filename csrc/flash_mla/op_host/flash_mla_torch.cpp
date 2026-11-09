// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <climits>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>
#include <tuple>
#include "torch_helper.h"
#include "flash_mla_host.h"
#include "aclrtlaunch_flash_mla.h"
#include "aclrtlaunch_flash_mla_fp16.h"
#include "../op_kernel/arch35/flash_mla_tiling_data.h"

namespace sglang::npu_kernel {
using Optional = c10::optional<at::Tensor>;
namespace {
using TilingKey=std::tuple<int,uintptr_t,std::string>;
struct TilingCache {
    std::mutex mutex;
    std::map<TilingKey,std::pair<at::Tensor,at::Tensor>> values;
};
TilingCache& Cache() {
    // Python clears this before NPU shutdown. Raw torch.ops users may keep it
    // until process exit, avoiding Tensor destruction after interpreter teardown.
    static auto* cache=new TilingCache;
    return *cache;
}
at::Tensor TilingTensor(const optiling::FlashMlaLaunchData& data, const at::Tensor& q) {
    // Exact content key avoids hash collisions. Each stream owns its initial H2D.
    auto& state=Cache();auto& cache=state.values;
    auto stream=c10_npu::getCurrentNPUStream().stream(false);
    TilingKey key{q.device().index(),reinterpret_cast<uintptr_t>(stream),
            std::string(reinterpret_cast<const char*>(&data),sizeof(data))};
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it=cache.find(key);
    if(it!=cache.end()) return it->second.second;
    auto cpu=at::empty({sizeof(data)},at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    std::memcpy(cpu.data_ptr(),&data,sizeof(data));
    auto gpu=cpu.to(q.device(),at::kByte,true,true);
    // Retain the pinned source: a first-use H2D captured in a graph replays it.
    cache.emplace(std::move(key),std::make_pair(cpu,gpu));
    return gpu;
}
void clear_flash_mla_cache() {
    auto& state=Cache();std::lock_guard<std::mutex> lock(state.mutex);state.values.clear();
}
void Check(const at::Tensor& t,const at::Tensor& q,at::ScalarType type,const char* name) {
    TORCH_CHECK(t.device()==q.device() && t.scalar_type()==type && t.is_contiguous(),
                name," must be contiguous, have the required dtype and be on the query device");
}

// Direct invocation of the single upstream-shaped template entry. Each specialization
// of the SDK-generated launch function owns the tiling key (template id * 1e6 + base
// key), so the key travels with the queued call and is never re-read from mutable
// state. BF16 uses the upstream entry name; FP16 keeps the same body compiled with
// ORIG_DTYPE_Q=DT_FLOAT16 and a distinct C entry name because both device binaries
// are linked into one host DSO.
template <uint8_t Layout, uint8_t KvLayout, bool HasMask>
inline void LaunchFlashMla(bool fp16, uint32_t blocks, void* stream, void* query, void* kCache, void* blockTable,
                           void* cacheSeqlens, void* cuSeqlensQ, void* sequsedQ, void* sinks, void* attnMask,
                           void* metadata, void* attnOut, void* softmaxLse, void* workspace, void* tiling) {
    if (fp16) {
        ACLRT_LAUNCH_KERNEL(flash_mla_fp16)<Layout, KvLayout, HasMask, 0>(
            blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, sinks, attnMask, metadata,
            attnOut, softmaxLse, workspace, tiling);
    } else {
        ACLRT_LAUNCH_KERNEL(flash_mla)<Layout, KvLayout, HasMask, 0>(
            blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, sinks, attnMask, metadata,
            attnOut, softmaxLse, workspace, tiling);
    }
}

template <uint8_t Layout>
inline void DispatchKv(bool fp16, int64_t kvLayout, bool hasMask, uint32_t blocks, void* stream, void* query,
                       void* kCache, void* blockTable, void* cacheSeqlens, void* cuSeqlensQ, void* sequsedQ,
                       void* sinks, void* attnMask, void* metadata, void* attnOut, void* softmaxLse,
                       void* workspace, void* tiling) {
    switch (kvLayout) {
        case 1:
            if (hasMask) LaunchFlashMla<Layout, 1, true>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                         cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                         softmaxLse, workspace, tiling);
            else LaunchFlashMla<Layout, 1, false>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                  cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                  softmaxLse, workspace, tiling);
            break;
        case 2:
            if (hasMask) LaunchFlashMla<Layout, 2, true>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                         cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                         softmaxLse, workspace, tiling);
            else LaunchFlashMla<Layout, 2, false>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                  cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                  softmaxLse, workspace, tiling);
            break;
        case 3:
            if (hasMask) LaunchFlashMla<Layout, 3, true>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                         cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                         softmaxLse, workspace, tiling);
            else LaunchFlashMla<Layout, 3, false>(fp16, blocks, stream, query, kCache, blockTable, cacheSeqlens,
                                                  cuSeqlensQ, sequsedQ, sinks, attnMask, metadata, attnOut,
                                                  softmaxLse, workspace, tiling);
            break;
        default:
            TORCH_CHECK(false, "unsupported FlashMLA KV layout");
    }
}

inline void DispatchFlashMla(bool fp16, int64_t layout, int64_t kvLayout, bool hasMask, uint32_t blocks,
                             void* stream, void* query, void* kCache, void* blockTable, void* cacheSeqlens,
                             void* cuSeqlensQ, void* sequsedQ, void* sinks, void* attnMask, void* metadata,
                             void* attnOut, void* softmaxLse, void* workspace, void* tiling) {
    switch (layout) {
        case 0:
            DispatchKv<0>(fp16, kvLayout, hasMask, blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ,
                          sequsedQ, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
            break;
        case 1:
            DispatchKv<1>(fp16, kvLayout, hasMask, blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ,
                          sequsedQ, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
            break;
        case 2:
            DispatchKv<2>(fp16, kvLayout, hasMask, blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ,
                          sequsedQ, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
            break;
        case 4:
            DispatchKv<4>(fp16, kvLayout, hasMask, blocks, stream, query, kCache, blockTable, cacheSeqlens, cuSeqlensQ,
                          sequsedQ, sinks, attnMask, metadata, attnOut, softmaxLse, workspace, tiling);
            break;
        default:
            TORCH_CHECK(false, "unsupported FlashMLA query layout");
    }
}
}

std::tuple<at::Tensor,at::Tensor> flash_mla(const at::Tensor& q,const at::Tensor& k,
    const Optional& bt,const Optional& seq,const Optional& cu,const Optional& used,
    const Optional& sinks,const Optional& mask,const Optional& metadata,int64_t dv,
    double scale,int64_t mode,int64_t left,int64_t right,int64_t maxq,int64_t maxkv,
    std::string layout,std::string kvlayout,std::string outlayout,bool lse) {
    TORCH_CHECK(q.device().type()==c10::DeviceType::PrivateUse1,"FlashMLA requires an NPU");
    const c10::DeviceGuard guard(q.device());
    TORCH_CHECK(q.scalar_type()==at::kBFloat16 || q.scalar_type()==at::kHalf,"q must be BF16 or FP16");
    TORCH_CHECK(q.is_contiguous() && q.size(-1)==576 && dv==512,"FlashMLA requires contiguous Dqk=576, Dv=512");
    TORCH_CHECK(layout=="BSND" || layout=="BNSD" || layout=="TND","invalid query layout");
    TORCH_CHECK(outlayout==layout || (layout=="TND" && outlayout=="NTD"),"invalid output layout");
    TORCH_CHECK(q.dim()==(layout=="TND"?3:4),"invalid query rank");
    TORCH_CHECK(kvlayout=="PA_BBND" || kvlayout=="PA_BNBD" || kvlayout=="PA_NZ","invalid KV layout");
    TORCH_CHECK(bt && seq,"block_table and cache_seqlens are required");
    TORCH_CHECK(k.device()==q.device() && k.scalar_type()==q.scalar_type(),"k_cache must match query dtype and device");
    Check(*bt,q,at::kInt,"block_table");
    Check(*seq,q,at::kInt,"cache_seqlens");if(metadata)Check(*metadata,q,at::kInt,"metadata");
    if(used) {Check(*used,q,at::kInt,"seqused_q");TORCH_CHECK(used->sizes()==seq->sizes(),"seqused_q shape mismatch");}
    TORCH_CHECK(bt->dim()==2 && seq->dim()==1 && bt->size(0)==seq->numel(),"invalid batch dimensions");
    TORCH_CHECK(mode==0 || mode==3 || mode==4,"mask_mode must be 0, 3 or 4");
    TORCH_CHECK(maxq>0 && maxkv>0,"positive maximum sequence lengths are required");
    int64_t batch=seq->numel(),heads=q.size(layout=="BNSD"?1:-2);
    int64_t sqlen=layout=="TND"?maxq:q.size(layout=="BNSD"?2:1);
    TORCH_CHECK(batch>0 && heads>0,"invalid batch or head count");
    if(layout=="TND") {
        TORCH_CHECK(cu && cu->dim()==1 && cu->numel()==batch+1,"TND requires cu_seqlens_q of length batch+1");
        Check(*cu,q,at::kInt,"cu_seqlens_q");
    } else {TORCH_CHECK(q.size(0)==batch && maxq>=sqlen,"query batch/sequence mismatch");}
    int64_t page=kvlayout=="PA_BBND"?k.size(1):(kvlayout=="PA_BNBD"?k.size(2):k.size(-2));
    TORCH_CHECK(page>=16 && page<=1024 && page%16==0,"page size must be a multiple of 16 in [16,1024]");
    TORCH_CHECK(bt->size(1)*page>=maxkv,"block table is smaller than max_seqlen_kv");
    if(kvlayout!="PA_NZ") {
        TORCH_CHECK(k.dim()==4 && k.size(-1)==576 && k.size(kvlayout=="PA_BBND"?2:1)==1,"invalid KV shape");
        TORCH_CHECK(k.stride(-1)==1 && k.stride(kvlayout=="PA_BBND"?1:2)==576 && k.stride(0)>=page*576,"KV rows must be contiguous; padding between blocks is supported");
    } else {
        TORCH_CHECK(k.dim()==5 && k.size(1)==1 && k.size(2)==36 && k.size(4)==16 && k.stride(4)==1 && k.stride(3)==16 && k.stride(2)==page*16 && k.stride(0)>=page*576,"invalid PA_NZ shape or strides");
    }
    if(mode) {TORCH_CHECK(mask && mask->dim()==2 && mask->size(0)==2048 && mask->size(1)==2048,"mask must be [2048,2048]");Check(*mask,q,at::kChar,"attn_mask");}
    const bool hasSinks=sinks && sinks->numel()>0;
    if(hasSinks) {Check(*sinks,q,at::kFloat,"sinks");TORCH_CHECK(sinks->numel()==heads,"sinks must have one entry per head");}

    const auto cores=FlashMlaCoreNums();
    const int64_t cubeCores=cores.first,vectorCores=cores.second;
    optiling::FlashMlaLaunchData t;std::memset(&t,0,sizeof(t));
    t.dtype=q.scalar_type()==at::kHalf;t.layout=layout=="BSND"?0:layout=="BNSD"?1:outlayout=="NTD"?4:2;
    t.kvLayout=kvlayout=="PA_BBND"?1:kvlayout=="PA_BNBD"?2:3;t.hasMask=mode!=0;
    at::Tensor schedule=metadata?*metadata:at::Tensor();
    if(!schedule.defined() || schedule.numel()==0)
        schedule=flash_mla_metadata_impl(*seq,heads,1,cu,used,maxq,maxkv,576,512,mode,left,right,layout);
    TORCH_CHECK(schedule.numel()>=16+32*16,"invalid FlashMLA metadata capacity");
    auto& base=t.baseTiling.flashMlaBaseParams;
    base.bSize=batch;base.t1Size=layout=="TND"?q.size(0):batch*sqlen;base.n2Size=1;
    base.gSize=heads;base.s1Size=sqlen;base.s2Size=maxkv;base.dSize=512;base.dSizeV=512;base.dSizeRope=64;
    base.actualSeqLengthsQSize=cu?cu->numel():0;base.actualSeqLengthsKVSize=batch;
    base.scaleValue=scale;base.isSoftMaxLseEnable=lse;base.coreNum=cubeCores;
    base.outputLayout=layout=="TND"?(outlayout=="NTD"?6:4):(layout=="BNSD"?3:1);
    base.l2CacheOffFlag=heads*sqlen<=128;
    base.keyStrides.bnStride=k.stride(0);base.keyStrides.n2Stride=k.stride(kvlayout=="PA_BBND"?2:1);
    base.valueStrides=base.keyStrides;base.kRopeStrides=base.keyStrides;
    auto& mp=t.baseTiling.flashMlaAttenMaskParams;mp.sparseMode=mode;
    mp.preTokens=mode==4 && left>=0?left:INT_MAX;mp.nextTokens=mode==3?0:mode==4 && right>=0?right:INT_MAX;
    if(mode) {mp.attenMaskBatch=1;mp.attenMaskS1Size=2048;mp.attenMaskS2Size=2048;}
    auto& pa=t.baseTiling.flashMlaPageAttentionParams;pa.blockSize=page;pa.maxBlockNumPerBatch=bt->size(1);
    pa.paLayoutType=kvlayout=="PA_BBND"?1:kvlayout=="PA_BNBD"?0:2;
    t.baseTiling.flashMlaSystemPrefixParams.isActualSharedPrefixLenNull=1;
    auto shape=q.sizes().vec();shape.back()=512;
    if(outlayout=="NTD")std::swap(shape[0],shape[1]);
    auto output=at::empty(shape,q.options());
    std::vector<int64_t> lseshape{0};
    if(lse)lseshape=layout=="TND"?std::vector<int64_t>{heads,q.size(0)}:std::vector<int64_t>{batch,heads,sqlen};
    auto softmax=at::empty(lseshape,q.options().dtype(at::kFloat));
    auto& ep=t.baseTiling.flashMlaEmptyTensorParams;ep.needInit=1;
    ep.totalOutputSize=output.numel();ep.totalSoftMaxLseOutputSize=softmax.numel();ep.singleCoreSize=(output.numel()+vectorCores-1)/vectorCores;
    auto& wp=t.baseTiling.flashMlaWorkspaceParams;wp.accumOutSize=cubeCores*2*96*512;wp.logSumExpSize=cubeCores*2*96*8;
    // Per-call allocation is stream-safe and retained by the graph memory pool.
    // AscendC's HAVE_WORKSPACE wrapper advances the user pointer by 16 MiB on arch35.
    constexpr int64_t systemWorkspaceBytes=16LL*1024*1024;
    auto workspace=at::empty({systemWorkspaceBytes+(wp.accumOutSize+2LL*wp.logSumExpSize)*4},q.options().dtype(at::kByte));
    auto tiling=TilingTensor(t,q);
    void* qp=q.data_ptr();void* kp=k.data_ptr();void* bp=bt->data_ptr();void* sp=seq->data_ptr();
    void* cp=cu?cu->data_ptr():nullptr;void* up=used?used->data_ptr():nullptr;void* sip=hasSinks?sinks->data_ptr():nullptr;
    void* ap=mode?mask->data_ptr():nullptr;void* mdp=schedule.defined()?schedule.data_ptr():nullptr;
    void* op=output.data_ptr();void* lp=softmax.data_ptr();void* wsp=workspace.data_ptr();void* tp=tiling.data_ptr();
    const uint32_t blocks=cubeCores;
    // Capture every dispatch value by value: the queued OpCommand callback can run
    // after this function returns, so the specialization must already be selected
    // from the per-call state, not from mutable thread-local tiling state.
    const bool fp16=t.dtype!=0;
    const int64_t layoutSel=t.layout;
    const int64_t kvLayoutSel=t.kvLayout;
    const bool maskSel=t.hasMask!=0;
    auto acl_stream=c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call=[fp16,layoutSel,kvLayoutSel,maskSel,blocks,acl_stream,qp,kp,bp,sp,cp,up,sip,ap,mdp,op,lp,wsp,tp]() -> int {
        DispatchFlashMla(fp16,layoutSel,kvLayoutSel,maskSel,blocks,acl_stream,qp,kp,bp,sp,cp,up,sip,ap,mdp,op,lp,wsp,tp);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi(fp16 ? "flash_mla_fp16" : "flash_mla",acl_call);
    return {output,softmax};
}
}
TORCH_LIBRARY_FRAGMENT(npu,m) {
 m.def("_flash_mla_clear_cache() -> ()");
 m.def("flash_mla(Tensor q, Tensor k_cache, Tensor? block_table=None, Tensor? cache_seqlens=None, Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, int head_dim_v=512, float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1, int max_seqlen_q=-1, int max_seqlen_kv=-1, str layout_q='BSND', str layout_kv='PA_BBND', str layout_out='BSND', bool return_softmax_lse=False) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("flash_mla",TORCH_FN(sglang::npu_kernel::flash_mla));}
TORCH_LIBRARY_IMPL(npu,CatchAll,m) {m.impl("_flash_mla_clear_cache",TORCH_FN(sglang::npu_kernel::clear_flash_mla_cache));}
