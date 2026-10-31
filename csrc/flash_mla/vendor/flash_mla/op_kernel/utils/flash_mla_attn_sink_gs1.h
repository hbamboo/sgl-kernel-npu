/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_attn_sink_gs1.h
 * \brief Attention sink copy and initialization.
 */
#ifndef ATTN_SINK_GS1_H_
#define ATTN_SINK_GS1_H_

#include "kernel_operator.h"
#include "../../../common/op_kernel/const_def.h"

using namespace AscendC;

namespace AttentionCommon {
using namespace Reg;

template <typename INPUT_T>
__aicore__ inline void SinkCopyInGS1(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start / actS1Size;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End / actS1Size;

    uint32_t gCount = gEnd - gStart + 1;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
    DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
}

template <typename INPUT_T>
__aicore__ inline void SinkCopyInS1G(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;

    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;

    if (s1Start == s1End) {
        uint32_t gCount = gEnd - gStart + 1;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
    } else if (s1Start + 1 == s1End) {
        if (gStart > gEnd) {
            // 尾段放 UB 起始，头段按 32B 对齐追加。
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = (gEnd + 1) * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);

            uint32_t ubOffset = AttentionCommon::Align(gEnd + 1, 8U);
            sinkCopyParams.blockLen = (gSize - gStart) * sizeof(INPUT_T);
            DataCopyPad(sinkTmpUb[ubOffset], sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);

        } else {
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
        }
    } else {
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
    }
}

template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkBrcVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t m)
{
    RegTensor<T> valReg;
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(valReg, inUb);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, valReg, pregF);
}

template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkContigDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t m)
{
    RegTensor<T> fReg;
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fReg, inUb);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, fReg, pregF);
}

template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkGs1DirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t s0, uint32_t actS1Size,
                                        uint32_t m)
{
    RegTensor<T> fRegA;
    RegTensor<T> fRegB;
    RegTensor<T> gRegA;
    RegTensor<T> gRegB;
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> qSub;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegA, inUb);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegB, inUb + 64);

    Reg::Arange<int32_t>(aReg, static_cast<int32_t>(s0));
    RegTensor<int32_t> aszReg;
    Reg::Duplicate<int32_t>(aszReg, static_cast<int32_t>(actS1Size), pregAll);
    Reg::Div<int32_t>(qReg, aReg, aszReg, pregAll);
    Reg::Adds<int32_t>(qSub, qReg, -64, pregAll);

    RegTensor<uint32_t>& qU32 = (RegTensor<uint32_t>&)qReg;
    RegTensor<uint32_t>& qSubU32 = (RegTensor<uint32_t>&)qSub;
    Reg::Gather<T>(gRegA, fRegA, qU32);
    Reg::Gather<T>(gRegB, fRegB, qSubU32);
    MaskReg pregA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregA, qReg, 64, pregAll);
    Reg::Select<T>(oReg, gRegA, gRegB, pregA);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkRotDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t gStart, uint32_t gSize,
                                        uint32_t m)
{
    RegTensor<T> fRegA;
    RegTensor<T> fRegB;
    RegTensor<T> gRegA;
    RegTensor<T> gRegB;
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> kReg;
    RegTensor<int32_t> kSub;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegA, inUb);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegB, inUb + 64);

    Reg::Arange<int32_t>(aReg, static_cast<int32_t>(gStart));
    RegTensor<int32_t> gszReg;
    Reg::Duplicate<int32_t>(gszReg, static_cast<int32_t>(gSize), pregAll);
    Reg::Div<int32_t>(qReg, aReg, gszReg, pregAll);
    Reg::Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
    Reg::Sub<int32_t>(kReg, aReg, kReg, pregAll);
    Reg::Adds<int32_t>(kSub, kReg, -64, pregAll);

    RegTensor<uint32_t>& kU32 = (RegTensor<uint32_t>&)kReg;
    RegTensor<uint32_t>& kSubU32 = (RegTensor<uint32_t>&)kSub;
    Reg::Gather<T>(gRegA, fRegA, kU32);
    Reg::Gather<T>(gRegB, fRegB, kSubU32);
    MaskReg pregA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregA, kReg, 64, pregAll);
    Reg::Select<T>(oReg, gRegA, gRegB, pregA);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkGapDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t headOff, uint32_t headLen,
                                        uint32_t m)
{
    RegTensor<T> fHeadA;
    RegTensor<T> fHeadB;
    RegTensor<T> fTail;
    RegTensor<T> gHeadA;
    RegTensor<T> gHeadB;
    RegTensor<T> gHead;
    RegTensor<T> gTail;
    RegTensor<T> oReg;
    RegTensor<int32_t> rReg;
    RegTensor<int32_t> rSub;
    RegTensor<int32_t> idxTail;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fHeadA, inUb + headOff);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fHeadB, inUb + headOff + 64);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fTail, inUb);

    Reg::Arange<int32_t>(rReg, 0);
    Reg::Adds<int32_t>(rSub, rReg, -64, pregAll);
    Reg::Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregAll);

    RegTensor<uint32_t>& rU32 = (RegTensor<uint32_t>&)rReg;
    RegTensor<uint32_t>& rSubU32 = (RegTensor<uint32_t>&)rSub;
    Reg::Gather<T>(gHeadA, fHeadA, rU32);
    Reg::Gather<T>(gHeadB, fHeadB, rSubU32);
    MaskReg pregHA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregHA, rReg, 64, pregAll);
    Reg::Select<T>(gHead, gHeadA, gHeadB, pregHA);

    RegTensor<uint32_t>& idxTailU32 = (RegTensor<uint32_t>&)idxTail;
    Reg::Gather<T>(gTail, fTail, idxTailU32);
    MaskReg pregHead = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregHead, rReg, static_cast<int32_t>(headLen), pregAll);
    Reg::Select<T>(oReg, gHead, gTail, pregHead);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

template <typename T, typename INPUT_T, bool IS_GS1_LAYOUT>
__aicore__ inline void SinkExpandMaxVf(const LocalTensor<T>& maxUb, const LocalTensor<INPUT_T>& sinkTmpUb,
                                       uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
{
    if (actVecMSize == 0) {
        return;
    }
    __ubuf__ T* maxUbPtr = (__ubuf__ T*)maxUb.GetPhyAddr();
    __ubuf__ INPUT_T* inUb = (__ubuf__ INPUT_T*)sinkTmpUb.GetPhyAddr();

    if constexpr (IS_GS1_LAYOUT) {
        uint32_t s0 = gs1Start % actS1Size;
        SinkGs1DirectVF<T, INPUT_T>(maxUbPtr, inUb, s0, actS1Size, actVecMSize);
        return;
    }
    if (gSize == 1U) {
        SinkBrcVF<T, INPUT_T>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;
    if (s1Start == s1End) {
        SinkContigDirectVF<T, INPUT_T>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    if (s1Start + 1 == s1End && gStart > gEnd) {
        uint32_t headOff = (gEnd + 1 + 7U) & ~7U;
        uint32_t headLen = gSize - gStart;
        SinkGapDirectVF<T, INPUT_T>(maxUbPtr, inUb, headOff, headLen, actVecMSize);
        return;
    }

    SinkRotDirectVF<T, INPUT_T>(maxUbPtr, inUb, gStart, gSize, actVecMSize);
}
} // namespace AttentionCommon

#endif // ATTN_SINK_GS1_H_
