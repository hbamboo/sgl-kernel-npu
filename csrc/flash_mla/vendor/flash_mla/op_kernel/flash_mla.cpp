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
 * \file flash_mla.cpp
 * \brief FlashMla kernel entry.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"

#include "arch35/flash_mla_template_tiling_key.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "adv_api/activation/softmax.h"
#include "utils/flash_mla_type.h"
using namespace optiling;
#include "arch35/flash_mla_kernel.h"

using namespace AscendC;

template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FLASH_MLA_LAYOUT GetQueryLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND) || (inOutLayoutType == InOutLayoutType_TND_NTD),
                  "GetQueryLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return FLASH_MLA_LAYOUT::BSH;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return FLASH_MLA_LAYOUT::BNSD;
    } else { // InOutLayoutType_TND / InOutLayoutType_TND_NTD
        return FLASH_MLA_LAYOUT::TND;
    }
}

template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FLASH_MLA_LAYOUT GetOutLayoutMla()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) || (inOutLayoutType == InOutLayoutType_BNSD) ||
                      (inOutLayoutType == InOutLayoutType_TND) || (inOutLayoutType == InOutLayoutType_TND_NTD),
                  "GetOutLayoutMla fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_TND_NTD) {
        return FLASH_MLA_LAYOUT::NTD;
    } else {
        return GetQueryLayoutMla<inOutLayoutType>();
    }
}

template <uint8_t KvLayoutType>
__aicore__ inline constexpr bool IsPageAttentionMla()
{
    static_assert(KvLayoutType <= KvLayoutType_PA_NZ, "IsPageAttentionMla fail, KvLayoutType is incorrect");
    return (KvLayoutType != KvLayoutType_NO_PA);
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType>
__aicore__ inline constexpr GmFormat GetKvLayoutMla()
{
    return FlashAttnKernel::GetKVGmFormat<GetQueryLayoutMla<inOutLayoutType>(), KvLayoutType,
                                          IsPageAttentionMla<KvLayoutType>()>();
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
__global__ __aicore__ void flash_mla(__gm__ uint8_t* query, __gm__ uint8_t* kCache, __gm__ uint8_t* blockTable,
                                     __gm__ uint8_t* cacheSeqlens, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                     __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
                                     __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                     __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashMlaTilingData);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (ORIG_DTYPE_Q == DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#endif

    constexpr FLASH_MLA_LAYOUT qLayout = GetQueryLayoutMla<inOutLayoutType>();
    constexpr FLASH_MLA_LAYOUT outLayout = GetOutLayoutMla<inOutLayoutType>();
    constexpr GmFormat kvGmFormat = GetKvLayoutMla<inOutLayoutType, KvLayoutType>();
    constexpr bool pageAttention = IsPageAttentionMla<KvLayoutType>();
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(96);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(112);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    // 静态图 tiling 可能是常量字节数组，须拷贝到局部对象后访问。
    GET_TILING_DATA_MEMBER(optiling::FlashMlaTilingData, baseTiling, baseTilingIn, tiling);
    const optiling::FlashMlaNoQuantTilingArch35* __restrict tilingData = &baseTilingIn;

    // 复位 SoC 状态，避免继承前序算子的原子累加和 Mask 设置。
    AscendC::InitSocState();

    using FA_T_Cube = FlashAttnKernel::FATypeCube<INPUT_T, float, qLayout, s1TemplateType, s2TemplateType,
                                                  dTemplateType, dVTemplateType, KvLayoutType>;
    using FA_T_Vec =
        FlashAttnKernel::FATypeVec<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                   dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;
    using FA_T_Fd = FlashAttnKernel::FATypeFd<INPUT_T, float, OUT_T, qLayout, outLayout, s1TemplateType, s2TemplateType,
                                              dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType>;

#ifdef __DAV_C310_CUBE__
    using CubeBlock = FlashAttnKernel::FlashMlaNoQuantMlaBlockCube<FA_T_Cube>;
    using VecFaBlock = FlashAttnKernel::FlashMlaNoQuantMlaBlockVecDummy<FA_T_Vec>;
    using VecFdBlock = FlashAttnKernel::FlashMlaNoQuantMlaBlockVecDummy<FA_T_Vec>;
#else
    using CubeBlock = FlashAttnKernel::FlashMlaNoQuantMlaBlockCubeDummy<FA_T_Cube>;
    using VecFaBlock = FlashAttnKernel::FlashMlaNoQuantMlaBlockVec<FA_T_Vec>;
    using VecFdBlock = FlashAttnKernel::FlashMlaBlockVecFlashDecodeMla<FA_T_Fd>;
#endif
    using Kernel = FlashAttnKernel::FlashAttentionNoQuantMlaKernel<CubeBlock, VecFaBlock, VecFdBlock>;

    Kernel op;
    op.Init(query, kCache, sinks, attnMask, cuSeqlensQ, sequsedQ, cacheSeqlens, blockTable, softmaxLse, attnOut, user,
            metadata, tilingData);
    op.Process();

    AscendC::PipeBarrier<PIPE_ALL>();
}
