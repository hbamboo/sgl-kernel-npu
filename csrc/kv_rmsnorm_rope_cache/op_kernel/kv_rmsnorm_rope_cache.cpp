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
 * \file kv_rmsnorm_rope_cache.cpp
 * \brief Named AscendC entry for the unified-576 latent-cache RMSNorm+RoPE update.
 */

#include "kernel_operator.h"
#include "kv_rmsnorm_rope_cache.h"
#include "kv_rmsnorm_rope_tiling.h"

using namespace AscendC;

namespace {
__aicore__ inline void CopyTilingData(KvRmsNormRopeTiling* dst, GM_ADDR src)
{
    auto* dstWords = reinterpret_cast<uint64_t*>(dst);
    const __gm__ uint64_t* srcWords = reinterpret_cast<__gm__ uint64_t*>(src);
    for (uint32_t i = 0; i < sizeof(KvRmsNormRopeTiling) / sizeof(uint64_t); ++i) {
        dstWords[i] = srcWords[i];
    }
}
} // namespace

extern "C" __global__ __aicore__ void sgl_kv_rmsnorm_rope_cache(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR cache, GM_ADDR kOut, GM_ADDR vOut,
    GM_ADDR tiling)
{
    // Retained verbatim from the upstream arch35 entry
    // (kv_rms_norm_rope_cache_apt.cpp): disable global overflow mode for the
    // RegBase arithmetic and restore the caller's setting afterwards.
#define GLOBAL_OVERFLOW_MODE_CTRL 60
#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif
    KvRmsNormRopeTiling tilingData;
    CopyTilingData(&tilingData, tiling);
    if (tilingData.dtypeFlag == 0) {
        KvRmsNormRopeCache::KvRmsNormRopeCacheKernel<bfloat16_t> op(&pipe, &tilingData);
        op.Init(kv, gamma, cos, sin, index, cache, kOut, vOut);
        op.Process();
    } else {
        KvRmsNormRopeCache::KvRmsNormRopeCacheKernel<half> op(&pipe, &tilingData);
        op.Init(kv, gamma, cos, sin, index, cache, kOut, vOut);
        op.Process();
    }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
}
