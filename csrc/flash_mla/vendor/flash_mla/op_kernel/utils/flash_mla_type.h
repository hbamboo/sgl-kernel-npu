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
 * \file flash_mla_type.h
 * \brief flash_mla 编译期模板参数聚合类型。
 */

#ifndef FLASH_MLA_TYPE_H_
#define FLASH_MLA_TYPE_H_

#include "flash_mla_common_def.h"
#include "flash_mla_memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../arch35/flash_mla_public_define_arch35.h"

namespace FlashAttnKernel {

template <FLASH_MLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <FLASH_MLA_LAYOUT LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

template <FLASH_MLA_LAYOUT LAYOUT>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT == FLASH_MLA_LAYOUT::TND || LAYOUT == FLASH_MLA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FLASH_MLA_LAYOUT LAYOUT, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT == FLASH_MLA_LAYOUT::TND || LAYOUT == FLASH_MLA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FLASH_MLA_LAYOUT LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT == FLASH_MLA_LAYOUT::BSH) || (LAYOUT == FLASH_MLA_LAYOUT::BNSD) ||
                      (LAYOUT == FLASH_MLA_LAYOUT::TND) || (LAYOUT == FLASH_MLA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BSH || LAYOUT == FLASH_MLA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT == FLASH_MLA_LAYOUT::BNSD || LAYOUT == FLASH_MLA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}

template <typename INPUT_T, typename T, FLASH_MLA_LAYOUT LAYOUT = FLASH_MLA_LAYOUT::NONE,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0>
struct FATypeCube {
    using inputType = INPUT_T;
    using mmType = T;
    static constexpr FLASH_MLA_LAYOUT qLayout = LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

template <
    typename INPUT_T, typename T, typename OUTPUT_T, FLASH_MLA_LAYOUT LAYOUT = FLASH_MLA_LAYOUT::NONE,
    FLASH_MLA_LAYOUT OUT_LAYOUT = FLASH_MLA_LAYOUT::NONE, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0>
struct FATypeVec {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr FLASH_MLA_LAYOUT qLayout = LAYOUT;
    static constexpr FLASH_MLA_LAYOUT attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
    static constexpr bool flashDecode = true;
};

template <
    typename INPUT_T, typename T, typename OUTPUT_T, FLASH_MLA_LAYOUT LAYOUT = FLASH_MLA_LAYOUT::NONE,
    FLASH_MLA_LAYOUT OUT_LAYOUT = FLASH_MLA_LAYOUT::NONE, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool HAS_MASK = false, uint8_t KvLayoutType = 0>
struct FATypeFd {
    using inputType = INPUT_T;
    using mmType = T;
    using outputType = OUTPUT_T;
    static constexpr FLASH_MLA_LAYOUT qLayout = LAYOUT;
    static constexpr FLASH_MLA_LAYOUT attnOutLayout = OUT_LAYOUT;
    static constexpr S1TemplateType mBaseSize = s1TemplateType;
    static constexpr S2TemplateType s2BaseSize = s2TemplateType;
    static constexpr DTemplateType dBaseSize = dTemplateType;
    static constexpr DTemplateType dVBaseSize = dVTemplateType;
    static constexpr bool hasMask = HAS_MASK;
    static constexpr uint8_t kvLayoutType = KvLayoutType;
    static constexpr bool pageAttention = (KvLayoutType > 0);
};

template <ActualSeqLensMode Q_MODE, ActualSeqLensMode KV_MODE>
class FlashMlaSeqLensTool {
public:
    using SEQLEN_T = uint32_t;

    GlobalTensor<SEQLEN_T> actualSeqLengthsGmQ;
    GlobalTensor<SEQLEN_T> actualSeqLengthsGm;

    static constexpr bool Q_WITH_ZERO_HEAD = (Q_MODE == ActualSeqLensMode::ACCUM);
    ActualSeqLensParser<Q_MODE, SEQLEN_T, Q_WITH_ZERO_HEAD> qActSeqLensParser;
    ActualSeqLensParser<KV_MODE, SEQLEN_T, false> kvActSeqLensParser;

    __aicore__ inline void InitQ(__gm__ uint8_t* cuSeqlensQAddr, __gm__ uint8_t* sequsedQAddr,
                                 uint32_t actualSeqLenSize, uint64_t s1Size, uint32_t batchSize)
    {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ SEQLEN_T*)cuSeqlensQAddr, actualSeqLenSize);
        if (sequsedQAddr == nullptr) {
            qActSeqLensParser.Init(actualSeqLengthsGmQ, actualSeqLenSize, s1Size);
        } else {
            if constexpr (Q_WITH_ZERO_HEAD) {
                qActSeqLensParser.Init((__gm__ uint8_t*)cuSeqlensQAddr, actualSeqLenSize, (__gm__ uint8_t*)sequsedQAddr,
                                       actualSeqLenSize - 1);
            } else {
                qActSeqLensParser.Init((__gm__ uint8_t*)sequsedQAddr, batchSize, s1Size);
            }
        }
    }

    __aicore__ inline void InitKv(__gm__ uint8_t* cacheSeqlensAddr, uint32_t actualSeqLenKVSize, uint64_t s2Size)
    {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ SEQLEN_T*)cacheSeqlensAddr, actualSeqLenKVSize);
        kvActSeqLensParser.Init(actualSeqLengthsGm, actualSeqLenKVSize, s2Size);
    }
};

} // namespace FlashAttnKernel

#endif // FLASH_MLA_TYPE_H_
