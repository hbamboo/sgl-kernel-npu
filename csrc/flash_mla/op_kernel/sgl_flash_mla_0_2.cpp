#include "kernel_operator.h"
#ifndef __ASC_NPU_HOST__
#include "flash_mla_impl.h"
#endif
extern "C" __global__ __aicore__ void sgl_flash_mla_0_2(GM_ADDR query, GM_ADDR kCache,
    GM_ADDR blockTable, GM_ADDR cacheSeqlens, GM_ADDR cuSeqlensQ, GM_ADDR sequsedQ,
    GM_ADDR sinks, GM_ADDR attnMask, GM_ADDR metadata, GM_ADDR attnOut,
    GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
#ifndef __ASC_NPU_HOST__
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(optiling::FlashMlaLaunchData);
    optiling::FlashMlaLaunchData t;
    optiling::CopyFlashMlaTiling(tiling,t);
    if(t.kvLayout==1 && t.hasMask==0) RunFlashMla<bfloat16_t,2,1,false>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
    if(t.kvLayout==1 && t.hasMask==1) RunFlashMla<bfloat16_t,2,1,true>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
    if(t.kvLayout==2 && t.hasMask==0) RunFlashMla<bfloat16_t,2,2,false>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
    if(t.kvLayout==2 && t.hasMask==1) RunFlashMla<bfloat16_t,2,2,true>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
    if(t.kvLayout==3 && t.hasMask==0) RunFlashMla<bfloat16_t,2,3,false>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
    if(t.kvLayout==3 && t.hasMask==1) RunFlashMla<bfloat16_t,2,3,true>(query,kCache,blockTable,cacheSeqlens,cuSeqlensQ,sequsedQ,sinks,attnMask,metadata,attnOut,softmaxLse,workspace,&t.baseTiling);
#endif
}
