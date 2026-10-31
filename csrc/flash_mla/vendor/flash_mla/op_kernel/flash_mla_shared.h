/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_SHARED_H
#define FLASH_MLA_SHARED_H

#include <cstdint>

namespace FlashMlaCommon {
constexpr int64_t MLA_KV_HEADS = 1;
constexpr int64_t MLA_HEAD_DIM_QK = 576;
constexpr int64_t MLA_HEAD_DIM_QK_NOPE = 512;
constexpr int64_t MLA_HEAD_DIM_V = 512;
constexpr uint32_t HEAD_METADATA_STRIDE = 16U;
constexpr uint32_t FA_METADATA_STRIDE = 16U;
constexpr uint32_t FD_METADATA_STRIDE = 16U;
} // namespace FlashMlaCommon

#endif // FLASH_MLA_SHARED_H
