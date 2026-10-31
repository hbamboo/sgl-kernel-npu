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
 * \file flash_mla_template_tiling_key.h
 * \brief FlashMla tiling keys for arch35.
 */

#ifndef FLASH_MLA_TEMPLATE_TILING_KEY_H_
#define FLASH_MLA_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#include "../utils/flash_mla_common_def.h"
#include "flash_mla_tiling_data.h"

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

// CANN 模板预编译器只解析数字 token，布尔取值须使用 0/1。
#define FLASH_MLA_TPL_BOOL_FALSE 0
#define FLASH_MLA_TPL_BOOL_TRUE 1

ASCENDC_TPL_ARGS_DECL(FlashMla, ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
                      ASCENDC_TPL_BOOL_DECL(HasAttenMask, FLASH_MLA_TPL_BOOL_FALSE, FLASH_MLA_TPL_BOOL_TRUE),
                      ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 0), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSND,
                                              InOutLayoutType_BNSD, InOutLayoutType_TND, InOutLayoutType_TND_NTD),
                         ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_BBND,
                                              KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ),
                         ASCENDC_TPL_BOOL_SEL(HasAttenMask, FLASH_MLA_TPL_BOOL_FALSE, FLASH_MLA_TPL_BOOL_TRUE),
                         ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST,
                                              Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FlashMlaTilingData)), );

#endif // FLASH_MLA_TEMPLATE_TILING_KEY_H_
