#include "scatter_pa_kv_cache_tiling_data.h"
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

template<class T,class I>
__aicore__ inline void RunScatter(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, const ScatterPaKvCacheTilingData& tilingData,uint64_t tilingKey) {
    AscendC::TPipe pipe;
    GM_ADDR key_cache_out=key_cache_in;
    GM_ADDR value_cache_out=value_cache_in;
    using keyType = typename GetInputType<T>::type;
    using valueType = typename GetInputType<T>::type;
    if (tilingKey == NORMAL_INT32_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NORMAL_INT32_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NORMAL_INT64_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NORMAL_INT64_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NORMAL_NON_CONTIGUOUS) {
        ScatterPaKvCache::ScatterPaKvCacheNormNonContiguous<keyType, I, DUAL_IN_OUT> op(
            &pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NZ_FULLY_LOAD) {
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
    } else if (tilingKey == NCT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalFullyLoad<keyType, I, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if (tilingKey == NCT_NOT_FULLY_LOAD) {
        ScatterPaKvCache::ScatterPaKvCacheNormalNotFullyLoad<keyType, I, DUAL_IN_OUT, true> op(
            &pipe, &tilingData);
        op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                seq_lens, key_cache_out, value_cache_out);
        op.Process();
    } else if constexpr (!(std::is_same<T, fp4x2_e2m1_t>::value ||
                           std::is_same<T, fp4x2_e1m2_t>::value)) {
        if (tilingKey == ROPE_INT32_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<T, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == ROPE_INT32_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<T, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == ROPE_INT64_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeFullyLoad<T, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == ROPE_INT64_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheRopeNotFullyLoad<T, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == OMNI_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheOmniFullyLoad<T, I, DUAL_IN_OUT> op(&pipe,
                                                                                                           &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == OMNI_NOT_FULLY_LOAD) {
            ScatterPaKvCache::ScatterPaKvCacheOmniNotFullyLoad<T, I, DUAL_IN_OUT> op(
                &pipe, &tilingData);
            op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                    seq_lens, key_cache_out, value_cache_out);
            op.Process();
        } else if (tilingKey == ALIBI_INT32_FULLY_LOAD) {
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int32_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if (tilingKey == ALIBI_INT32_NOT_FULLY_LOAD) {
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int32_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if (tilingKey == ALIBI_INT64_FULLY_LOAD) {
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int8_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int16_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiFullyLoad<int32_t, int64_t, DUAL_IN_OUT> op(&pipe, &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        } else if (tilingKey == ALIBI_INT64_NOT_FULLY_LOAD) {
            if constexpr (sizeof(T) == sizeof(int8_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int8_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                     &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int16_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int16_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            } else if constexpr (sizeof(T) == sizeof(int32_t)) {
                ScatterPaKvCache::ScatterPaKvCacheAlibiNotFullyLoad<int32_t, int64_t, DUAL_IN_OUT> op(&pipe,
                                                                                                      &tilingData);
                op.Init(key, key_cache_in, slot_mapping, value, value_cache_in, compress_lens, compress_seq_offset,
                        seq_lens, key_cache_out, value_cache_out);
                op.Process();
            }
        }
    }
}
extern "C" __global__ __aicore__ void sgl_scatter_pa_kv_cache(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens, GM_ADDR compress_seq_offset, GM_ADDR seq_lens, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScatterPaKvCacheLaunchData p;
    auto* dst=reinterpret_cast<uint64_t*>(&p);
    auto* src=reinterpret_cast<__gm__ uint64_t*>(tiling);
    for (int i=0;i<sizeof(p)/sizeof(uint64_t);++i) dst[i]=src[i];
    if (p.dtype==12) { if(p.indexBytes==4) RunScatter<bfloat16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<bfloat16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
    if (p.dtype==1) { if(p.indexBytes==4) RunScatter<half,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<half,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
    if (p.dtype==0) { if(p.indexBytes==4) RunScatter<float,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<float,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
    if (p.dtype==2) { if(p.indexBytes==4) RunScatter<int8_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<int8_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
    if (p.dtype==6) { if(p.indexBytes==4) RunScatter<int16_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<int16_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
    if (p.dtype==3) { if(p.indexBytes==4) RunScatter<int32_t,int32_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); else RunScatter<int32_t,int64_t>(key,key_cache_in,slot_mapping,value,value_cache_in,compress_lens,compress_seq_offset,seq_lens,p.data,p.key); }
}
