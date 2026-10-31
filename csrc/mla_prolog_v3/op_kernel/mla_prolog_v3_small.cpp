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
 * \file mla_prolog_v3.cpp
 * \brief
 */

#include "mla_prolog_v3_body.h"

extern "C" __global__ __aicore__ void mla_prolog_v3_small(
    __gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr, __gm__ uint8_t* weightUk,
    __gm__ uint8_t* weightDkvKr, __gm__ uint8_t* rmsnormGammaCq, __gm__ uint8_t* rmsnormGammaCkv,
    __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos, __gm__ uint8_t* kvCache, __gm__ uint8_t* krCache,
    __gm__ uint8_t* cacheIndex, __gm__ uint8_t* dequantScaleX, __gm__ uint8_t* dequantScaleWDq,
    __gm__ uint8_t* dequantScaleWUqQr, __gm__ uint8_t* dequantScaleWDkvKr, __gm__ uint8_t* quantScaleCkv,
    __gm__ uint8_t* quantScaleCkr, __gm__ uint8_t* smoothScalesCq, __gm__ uint8_t* actualSeqLen,
    __gm__ uint8_t* kNopeClipAlpha, __gm__ uint8_t* queryOut, __gm__ uint8_t* queryRopeOut, __gm__ uint8_t* kvCacheOut,
    __gm__ uint8_t* krCacheOut, __gm__ uint8_t* dequantScaleQNopeOut, __gm__ uint8_t* queryNormOut,
    __gm__ uint8_t* dequantScaleQNormOut, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    const uint64_t key = reinterpret_cast<const __gm__ optiling::MlaPrologTilingData *>(tiling)->tilingKey;
    switch (static_cast<uint32_t>(key >> 32)) {
        case 1: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 1, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 2: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 2, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 3: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 3, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 4: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 4, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 5: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 5, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 6: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 6, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 7: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 7, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 8: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 8, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 9: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 9, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 10: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 10, true, false>(MLA_PROLOG_V3_ARGS); break;
        case 17: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 1, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 18: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 2, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 19: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 3, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 20: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 4, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 21: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 5, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 22: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 6, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 23: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 7, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 24: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 8, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 25: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 9, true, true>(MLA_PROLOG_V3_ARGS); break;
        case 26: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 6, 10, true, true>(MLA_PROLOG_V3_ARGS); break;
    }
}
