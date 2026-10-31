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

extern "C" __global__ __aicore__ void mla_prolog_v3(
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
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    const uint64_t key = reinterpret_cast<const __gm__ optiling::MlaPrologTilingData *>(tiling)->tilingKey;

    if (const uint32_t staticTokens = key >> 32) {
        switch (staticTokens) {
            case 1: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 1>(MLA_PROLOG_V3_ARGS); break;
            case 2: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 2>(MLA_PROLOG_V3_ARGS); break;
            case 3: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 3>(MLA_PROLOG_V3_ARGS); break;
            case 4: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 4>(MLA_PROLOG_V3_ARGS); break;
            case 5: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 5>(MLA_PROLOG_V3_ARGS); break;
            case 6: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 6>(MLA_PROLOG_V3_ARGS); break;
            case 7: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 7>(MLA_PROLOG_V3_ARGS); break;
            case 8: MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7, 8>(MLA_PROLOG_V3_ARGS); break;
        }
        return;
    }
    switch (key) {
        // ---- bf16 / no quant, PA_BSND, 1:2 (golden / accuracy baseline)
        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 1, 0, 0, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 1, 0, false, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;

        // ---- MXFP8 weights (GLM-4.7-Flash), bf16 kv cache, PA_BSND, 1:2
        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 1, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, true, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 0, 0, 0, 0, 0, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, false, false, 0, 0, 0, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        case MLA_PROLOG_V3_KEY(1, 2, 7, 0, 0, 0, 0, 1, 1, 7):
            MlaPrologV3KernelBody<1, 2, 7, false, false, 0, 0, 1, true, 7>(MLA_PROLOG_V3_ARGS);
            break;
        default:
            // Unsupported combination for this build: nothing is written.
            // The host side checks the same key set before launching.
            break;
    }
}
