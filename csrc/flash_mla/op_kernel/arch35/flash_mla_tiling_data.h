/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_tiling_data.h
 * \brief flash_mla arch35 kernel 侧 tiling struct 与 metadata 消费常量
 */

#ifndef FLASH_MLA_TILING_DATA_H_
#define FLASH_MLA_TILING_DATA_H_

#include "../flash_mla_shared.h"

namespace optiling {

constexpr uint32_t FA_AIC_CORE_NUM = 36;
constexpr uint32_t FA_AIV_CORE_NUM = 72;

constexpr uint32_t FA_METADATA_HEADER_OFFSET = FlashMlaCommon::HEAD_METADATA_STRIDE * sizeof(uint32_t);
constexpr uint32_t FLASH_MLA_METADATA_AIC_NUM_INDEX = 4;
constexpr uint32_t FLASH_MLA_METADATA_AIV_NUM_INDEX = 5;

constexpr uint32_t FLASH_ATTN_BN2_START_INDEX = 0;
constexpr uint32_t FLASH_ATTN_M_START_INDEX = 1;
constexpr uint32_t FLASH_ATTN_S2_START_INDEX = 2;
constexpr uint32_t FLASH_ATTN_BN2_END_INDEX = 3;
constexpr uint32_t FLASH_ATTN_M_END_INDEX = 4;
constexpr uint32_t FLASH_ATTN_S2_END_INDEX = 5;
constexpr uint32_t FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

constexpr uint32_t FA_FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FA_FD_M_IDX_INDEX = 1;
constexpr uint32_t FA_FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FA_FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FA_FD_M_START_INDEX = 4;
constexpr uint32_t FA_FD_M_NUM_INDEX = 5;

struct stridesParams {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;

    void set_bnStride(uint64_t bnStride)
    {
        this->bnStride = bnStride;
    }
    uint64_t get_bnStride() const
    {
        return bnStride;
    }
    void set_n2Stride(uint64_t n2Stride)
    {
        this->n2Stride = n2Stride;
    }
    uint64_t get_n2Stride() const
    {
        return n2Stride;
    }
};

struct FlashMlaBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t t2Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t dSizeRope = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float scaleValue = 0.0f;
    uint8_t isKvContinuous = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint8_t l2CacheOffFlag = 0;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
    stridesParams keyStrides;
    stridesParams valueStrides;
    stridesParams kRopeStrides;
    stridesParams kScaleStrides;
    stridesParams vScaleStrides;
};

struct FlashMlaAttenMaskParams {
    uint8_t sparseMode = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint8_t isRowInvalidOpen = 0;
    uint8_t isExistRowInvalid = 0;
};

struct FlashMlaPseParams {
    uint8_t pseShiftByBatch = 0;
    uint32_t pseS1Size = 0;
    uint32_t pseS2Size = 0;
    uint32_t pseStride = 0;
    uint32_t qStartIdx = 0;
    uint32_t kvStartIdx = 0;
};

struct FlashMlaSystemPrefixParams {
    uint8_t isActualSharedPrefixLenNull = 0;
    uint32_t prefixSeqInnerSize = 0;
};

struct FlashMlaPageAttentionParams {
    uint8_t paLayoutType = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
};

struct FlashMlaLeftPaddingParams {
    uint8_t isQHasLeftPadding = 0;
    uint8_t isKVHasLeftPadding = 0;
};

struct FlashMlaPostQuantParams {
    uint8_t isPostQuantPerChnl = 0;
    uint8_t isPostQuantBF16 = 0;
};

struct FlashMlaWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct FlashMlaEmptyTensorParams {
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint64_t totalOutputSize = 0;
    uint64_t totalSoftMaxLseOutputSize = 0;
};

class FlashMlaNoQuantTilingArch35 {
public:
    FlashMlaBaseParams flashMlaBaseParams;
    FlashMlaAttenMaskParams flashMlaAttenMaskParams;
    FlashMlaPseParams flashMlaPseParams;
    FlashMlaSystemPrefixParams flashMlaSystemPrefixParams;
    FlashMlaPageAttentionParams flashMlaPageAttentionParams;
    FlashMlaLeftPaddingParams flashMlaLeftPaddingParams;
    FlashMlaPostQuantParams flashMlaPostQuantParams;
    FlashMlaWorkspaceParams flashMlaWorkspaceParams;
    FlashMlaEmptyTensorParams flashMlaEmptyTensorParams;
};

class FlashMlaTilingData {
public:
    FlashMlaNoQuantTilingArch35 baseTiling;
};

struct FlashMlaLaunchData {
    FlashMlaNoQuantTilingArch35 baseTiling;
    uint32_t dtype = 0, layout = 0, kvLayout = 1, hasMask = 0;
};

} // namespace optiling
#endif // FLASH_MLA_TILING_DATA_H_
