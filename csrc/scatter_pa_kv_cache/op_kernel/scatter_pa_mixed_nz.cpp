#include "scatter_pa_kv_cache_tiling_data.h"
/**
聽* Copyright (c) 2025 Huawei Technologies Co., Ltd.
聽* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
聽* CANN Open Software License Agreement Version 2.0 (the "License").
聽* Please refer to the License for details. You may not use this file except in compliance with the License.
聽* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
聽* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
聽* See LICENSE in the root of the software repository for the full text of the License.
聽*/

/*!
 * \file scatter_pa_kv_cache.cpp
 * \brief scatter_pa_kv_cache
 */

#include "arch35/scatter_pa_kv_cache_normal_fully_load.h"
#include "arch35/scatter_pa_kv_cache_normal_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_rope_fully_load.h"
#include "arch35/scatter_pa_kv_cache_rope_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_alibi_fully_load.h"
#include "arch35/scatter_pa_kv_cache_alibi_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_omni_fully_load.h"
#include "arch35/scatter_pa_kv_cache_omni_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_nz_fully_load.h"
#include "arch35/scatter_pa_kv_cache_nz_not_fully_load.h"
#include "arch35/scatter_pa_kv_cache_norm_non_contiguous.h"
#include "arch35/scatter_pa_kv_cache_nz_non_contiguous.h"
#include "arch35/common.h"

#define NORMAL_INT32_FULLY_LOAD 141
#define NORMAL_INT32_NOT_FULLY_LOAD 140

#define NORMAL_INT64_FULLY_LOAD 181
#define NORMAL_INT64_NOT_FULLY_LOAD 180

#define ROPE_INT32_FULLY_LOAD 241
#define ROPE_INT32_NOT_FULLY_LOAD 240

#define ROPE_INT64_FULLY_LOAD 281
#define ROPE_INT64_NOT_FULLY_LOAD 280

#define ALIBI_INT32_FULLY_LOAD 341
#define ALIBI_INT32_NOT_FULLY_LOAD 340

#define ALIBI_INT64_FULLY_LOAD 381
#define ALIBI_INT64_NOT_FULLY_LOAD 380

#define OMNI_FULLY_LOAD 401
#define OMNI_NOT_FULLY_LOAD 400

#define NZ_FULLY_LOAD 501
#define NZ_NOT_FULLY_LOAD 500

#define NCT_FULLY_LOAD 601
#define NCT_NOT_FULLY_LOAD 600

#define NORMAL_NON_CONTIGUOUS 700
#define NZ_NON_CONTIGUOUS 800
#define NZ_NON_CONTIGUOUS_FIVE_DIM 900

using namespace ScatterPaKvCache;

template <typename T>
struct GetInputType {
    using type = typename std::conditional<
        std::is_same<T, fp4x2_e1m2_t>::value, int8_t,
        typename std::conditional<std::is_same<T, fp4x2_e2m1_t>::value, int8_t,
            typename std::conditional<sizeof(T) == B8, int8_t,
                typename std::conditional<sizeof(T) == B16, int16_t,
                    typename std::conditional<sizeof(T) == B32, int32_t, T>::type>::type>::type>::type>::type;
};

template<class keyType,class valueType,class I>
__aicore__ inline void RunMixedNz(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, const ScatterPaKvCacheTilingData& tilingData,uint64_t tilingKey) {
    AscendC::TPipe pipe;
    GM_ADDR key_cache_out=key_cache_in;
    GM_ADDR value_cache_out=value_cache_in;
if (tilingKey == NZ_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNzFullyLoad<keyType, valueType, I, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NZ_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNzNotFullyLoad<keyType, valueType, I, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NZ_NON_CONTIGUOUS) {
        ScatterPaKvCache::ScatterPaKvCacheNzNonContiguous<keyType, valueType, I, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    }  else if (tilingKey == NZ_NON_CONTIGUOUS_FIVE_DIM) {
        ScatterPaKvCache::ScatterPaKvCacheNzNonContiguous<keyType, valueType, I, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    }
}
extern "C" __global__ __aicore__ void sgl_scatter_pa_mixed_nz(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScatterPaKvCacheLaunchData p;
    auto* dst=reinterpret_cast<uint64_t*>(&p);
    auto* src=reinterpret_cast<__gm__ uint64_t*>(tiling);
    for (int i=0;i<sizeof(p)/sizeof(uint64_t);++i) dst[i]=src[i];
auto kb=p.dtype & 0xffffffff; auto vb=p.dtype >> 32;
if(kb==1&&vb==2) { if(p.indexBytes==4) RunMixedNz<int8_t,int16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int8_t,int16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==1&&vb==4) { if(p.indexBytes==4) RunMixedNz<int8_t,int32_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int8_t,int32_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==2&&vb==1) { if(p.indexBytes==4) RunMixedNz<int16_t,int8_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int16_t,int8_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==2&&vb==4) { if(p.indexBytes==4) RunMixedNz<int16_t,int32_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int16_t,int32_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==4&&vb==1) { if(p.indexBytes==4) RunMixedNz<int32_t,int8_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int32_t,int8_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==4&&vb==2) { if(p.indexBytes==4) RunMixedNz<int32_t,int16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int32_t,int16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==1&&vb==1) { if(p.indexBytes==4) RunMixedNz<int8_t,int8_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int8_t,int8_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==2&&vb==2) { if(p.indexBytes==4) RunMixedNz<int16_t,int16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int16_t,int16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
if(kb==4&&vb==4) { if(p.indexBytes==4) RunMixedNz<int32_t,int32_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunMixedNz<int32_t,int32_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
}
