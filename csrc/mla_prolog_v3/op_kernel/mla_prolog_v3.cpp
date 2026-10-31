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

#define MLA_PROLOG_VERSION 3
#define GLOBAL_OVERFLOW_MODE_CTRL 60

// [sgl-kernel-npu port] must come first: skips the ASCENDC_TPL declarations that
// do not compile during the host pass of the direct-launch build.
#include "mla_prolog_v3_kernel_preamble.h"
#include <cstdio>

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
#include "../../mla_prolog/op_kernel/arch92/kernel_mla_prolog_split_n_arch92.h"
#include "../../mla_prolog/op_kernel/arch92/kernel_mla_prolog_split_m_arch92.h"
#elif __CCE_AICORE__ == 310
#include "mla_prolog_v3_tiling_copy.h"
#include "../../mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h"
#include "../../mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_m.h"
#else
#include "../../mla_prolog/op_kernel/arch22/kernel_mla_prolog_split_n_arch22.h"
#endif
using namespace MlaProlog;

template <uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode, bool EnableDequantOpt, bool EnableGroupComputeOpt,
          uint8_t EmptyTensorMode, uint8_t ActualSeqLenMode, uint8_t SplitMMode, bool EnableRope, uint8_t CvMode, uint32_t StaticTokens = 0>
__aicore__ inline void MlaPrologV3KernelBody(
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
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

    REGISTER_TILING_DEFAULT(optiling::MlaPrologTilingData);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    constexpr auto emptyMode = static_cast<EMPTY_TENSOR_MODE>(EmptyTensorMode);
    if constexpr (emptyMode == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        return;
    }
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
    constexpr auto actualSeqLenMode = static_cast<ACTUAL_SEQ_MODE>(ActualSeqLenMode);
    constexpr auto splitMMode = static_cast<SPLIT_M_MODE>(SplitMMode);
    constexpr uint32_t cvRatio = CvMode == ASCENDC_TPL_MIX_AIC_1_1 ? 1 : 2;

    // [sgl-kernel-npu port] the direct-launch build has no GET_TILING_DATA_WITH_STRUCT;
    // copy the GM tiling blob into local memory explicitly.
    optiling::MlaPrologTilingData tilingDataIn;
    optiling::CopyMlaPrologTilingDataFromGm(tiling, tilingDataIn);
    if constexpr (StaticTokens != 0) {
        const uint32_t tokens = StaticTokens;
        tilingDataIn.baseParams.tokenSize = tokens;
        tilingDataIn.baseParams.stepBatchSize = tokens;
        tilingDataIn.baseParams.vectorBlockNum = tokens;
        tilingDataIn.baseParams.headSizeX = 2048;
        tilingDataIn.baseParams.headSizeCq = 768;
        tilingDataIn.baseParams.headSizeCkv = 512;
        tilingDataIn.baseParams.headSizeQc = 3840;
        tilingDataIn.baseParams.headSizeQr = 1280;
        tilingDataIn.baseParams.headSizeKr = 64;
        tilingDataIn.baseParams.numHeadSize = 20;
        tilingDataIn.baseParams.numHeadKvSize = 1;
        tilingDataIn.baseParams.dimHeadSizeQc = 192;
        tilingDataIn.baseParams.dimHeadRope = 64;
        tilingDataIn.baseParams.mm1BlockNum = 16;
        tilingDataIn.baseParams.mm1SingleCoreN = 48;
        tilingDataIn.baseParams.mm2BlockNum = 9;
        tilingDataIn.baseParams.mm2SingleCoreN = 64;
        tilingDataIn.baseParams.mm3BlockNum = 16;
        tilingDataIn.baseParams.mm3SingleCoreN = 320;
        tilingDataIn.baseParams.mm4BlockNum = 10;
        tilingDataIn.baseParams.mm4SingleCoreBatch = 2;
        tilingDataIn.baseParams.queryNormFlag = 1;
        tilingDataIn.baseParams.queryHeadStride = 576;
        tilingDataIn.baseParams.queryRopeHeadStride = 576;
        tilingDataIn.baseParams.decodeHeadLocal = 0;
        tilingDataIn.baseParams.decodeParallelDown = 0;
        tilingDataIn.baseParams.decodeKvBlockStart = 0;
    }
    const optiling::MlaPrologTilingData* __restrict tilingData = nullptr;
    const optiling::MlaPrologBaseParams* __restrict tilingDataBaseParams = &tilingDataIn.baseParams;

    TPipe pipe;
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif
    if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, bfloat16_t, bfloat16_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, bfloat16_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, int8_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, bfloat16_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, int8_t, float, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                                     emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PERTILE) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, int8_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT_PERTILE) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, int8_t, float, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                                     emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    }
#if (__CCE_AICORE__ == 310) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                       static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, FP8E8M0, cacheMode, EnableDequantOpt,
                                       EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<
                MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, FP8E8M0, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                         emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode, EnableDequantOpt,
                                       EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<
                MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                         emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TILE) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode, EnableDequantOpt,
                                       EnableGroupComputeOpt, emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode, EnableDequantOpt,
                                         EnableGroupComputeOpt, emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
                op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, bfloat16_t, float, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                                     emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, HIF8, float, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                                     emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TILE) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, float, cacheMode, EnableDequantOpt,
                                     EnableGroupComputeOpt, emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TILE) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, HIF8, float, cacheMode, EnableDequantOpt, EnableGroupComputeOpt,
                                     emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>>
            op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
                cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    }
#endif
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
}

// ---------------------------------------------------------------------------
// [sgl-kernel-npu port] single named kernel entry, dispatch on the tiling key
// that the host tiling recorded in the tiling data.
//
// key layout (see csrc/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h and
// AscendC::FastEncodeTilingKeyDirect):
//   bit 0-3   CACHE_MODE          1=PA_BSND 2=PA_NZ 0=BSND 3/4=PA_BLK
//   bit 4-5   SCENARIO            1=BF16 no-quant 2=quant
//   bit 6-9   QUANT_MODE          QUANT_MODE enum value
//   bit 10    ENABLE_DEQUANT_OPT
//   bit 11    ENABLE_GROUP_COMPUTE_OPT
//   bit 12-13 EMPTY_TENSOR_MODE
//   bit 14-15 ACTUAL_SEQ_LEN_MODE
//   bit 16-17 SPLIT_M_MODE
//   bit 18    ENABLE_ROPE
//   bit 19-26 CV_MODE             7 = MIX_AIC_1_2 (1:2), 6 = MIX_AIC_1_1 (1:1)
// ---------------------------------------------------------------------------
#define MLA_PROLOG_V3_KEY(cacheMode, scenario, quantMode, dequantOpt, groupOpt, emptyMode, seqLenMode, splitMMode,   \
                          enableRope, cvMode)                                                                       \
    ((static_cast<uint64_t>(cacheMode)) | (static_cast<uint64_t>(scenario) << 4) |                                  \
     (static_cast<uint64_t>(quantMode) << 6) | (static_cast<uint64_t>(dequantOpt) << 10) |                          \
     (static_cast<uint64_t>(groupOpt) << 11) | (static_cast<uint64_t>(emptyMode) << 12) |                           \
     (static_cast<uint64_t>(seqLenMode) << 14) | (static_cast<uint64_t>(splitMMode) << 16) |                        \
     (static_cast<uint64_t>(enableRope) << 18) | (static_cast<uint64_t>(cvMode) << 19))

#define MLA_PROLOG_V3_ARGS                                                                                          \
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, \
        krCache, cacheIndex, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv,   \
        quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, kvCacheOut, krCacheOut, \
        dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace, tiling

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
