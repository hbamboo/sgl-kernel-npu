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
 * \file kv_rms_norm_rope_cache_regbase_base.h
 * \brief RegBase (DAV_3510) RMSNorm helpers copied from ops-transformer
 *        posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/
 *        kv_rms_norm_rope_cache_regbase_base.h (source 9.1.0 / 6f0d3ec6).
 *
 * This is a trimmed copy: only the non-quantized b16 RMSNorm helpers used by the
 * unified-576 cache kernel are retained. The asymmetric/symmetric int8/fp8/hifloat8
 * quantization paths (RmsNormAsymQuant*, RmsNormSymQuant*, CAST_*_TO_INT8/HIFLOAT8/
 * FLOAT8 traits) are deleted because the migrated route is explicitly limited to
 * BF16/FP16 non-quantized latent512/rope64 (DESIGN.md section 6). No arithmetic used
 * by the retained path was changed.
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H

#include "kernel_operator.h"
#include "platform.h"
#include "kv_rmsnorm_rope_tiling.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

// Host and kernel must agree on the vector width: the host sizes the RoPE
// workspace from KV_RMSNORM_VL_FP32, while the retained upstream RegBase code
// (RopeBasicComputeVF writes two full VL_FP32 vectors) uses platform::GetVRegSize().
static_assert(VL_FP32 == static_cast<uint32_t>(KV_RMSNORM_VL_FP32),
              "platform::GetVRegSize must be 256 B (VL_FP32=64) for this kernel");

constexpr static AscendC::Reg::CastTrait CAST_B16_TO_B32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::Reg::CastTrait CAST_FP32_TO_FP16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr int64_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_COUNT_SINGLE = 1;
constexpr int32_t BUFFER_COUNT_DOUBLE = 2;
constexpr int32_t COS_SIN_CHANNEL_COUNT = 4;
constexpr int32_t WORKSPACE_FLOAT_VECTOR_COUNT = 2;

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIXTY_THREE = 63;

template <typename T_KV>
class KvRmsNormRopeCacheRegbase
{
public:
    __aicore__ inline KvRmsNormRopeCacheRegbase()
    {}

protected:
    /* global memory address */
    GlobalTensor<T_KV> kvGm;
    GlobalTensor<T_KV> gammaGm;
    GlobalTensor<T_KV> cosGm;
    GlobalTensor<T_KV> sinGm;
    GlobalTensor<int64_t> indexGm;
    GlobalTensor<T_KV> kCacheGm;
    GlobalTensor<T_KV> vCacheGm;
    GlobalTensor<T_KV> kOutGm, vOutGm;

    /* variable */
    float epsilon = 1e-5;
    float reciprocal = 1.0;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t useCoreNum = GetBlockNum();

    int64_t ubFactor = 1;
    int64_t ubLoop = 1;
    int64_t ubTail = 1;

    /* tilingdata */
    int64_t dv = 0;
    int64_t dvAlign = 0;
    int64_t dvB8Align = 0;
    int64_t dkAlign = 0;
    int64_t dk = 0;
    int64_t dvLoopCount = 0;

protected:
    __aicore__ inline void RmsNormBasicComputeVF(
        const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
        int64_t aSize, int64_t& foldPoint, int64_t& outerLoopDstStride);
    __aicore__ inline void RmsNormPostVF(
        const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
        const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride);
    __aicore__ inline void RmsNormVF(
        const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
        const LocalTensor<float>& wsLocal, int64_t calcRow);
    __aicore__ inline void RmsNormPostWithMul(
        const LocalTensor<float>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& grammaTensor,
        const int64_t aSize, const int64_t rSize, const int64_t stride);
};

__aicore__ inline int64_t FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - ScalarCountLeadingZero(num);
        return (CONST_ONE << pow);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(
    __local_mem__ T* input, AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        AscendC::Reg::RegTensor<half> xFp16;
        DataCopy<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
        Cast<float, half, CAST_B16_TO_B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        AscendC::Reg::RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
            xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, CAST_B16_TO_B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeTOut(
    __local_mem__ T* dst, AscendC::Reg::RegTensor<float>& src, AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::Reg::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::Reg::RegTensor<T> xFp16;
        Cast<T, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

// Store unaligned float32 register data to UB. Retains the b16/float branches of the
// upstream helper; the int8/hifloat8/fp8 quantization branches are intentionally dropped.
template <typename T>
__aicore__ inline void StoreUnAlignOneTensor(
    __local_mem__ T*& output, Reg::RegTensor<float>& src, Reg::UnalignReg& uValue, Reg::MaskReg& preg,
    uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::RegTensor<half> xFp16Pack;
        Cast<half, float, CAST_FP32_TO_FP16>(xFp16, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)xFp16Pack, (Reg::RegTensor<uint32_t>&)xFp16);
        DataCopyUnAlign(output, xFp16Pack, uValue, postUpdateStride);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::RegTensor<bfloat16_t> xBf16Pack;
        Cast<bfloat16_t, float, CAST_FP32_TO_FP16>(xBf16, src, preg);
        Pack((Reg::RegTensor<uint16_t>&)xBf16Pack, (Reg::RegTensor<uint32_t>&)xBf16);
        DataCopyUnAlign(output, xBf16Pack, uValue, postUpdateStride);
    } else {
        DataCopyUnAlign(output, src, uValue, postUpdateStride);
    }
}

/*
gm->ub: lineNum rows, srcLineLength is the GM row length, copyLineLength the useful part.
*/
template <typename T>
__aicore__ inline void CopyInLineAlign(
    const LocalTensor<T>& dstUb, const GlobalTensor<T>& srcGm, int64_t lineNum, int64_t srcLineLength,
    int64_t copyLineLength)
{
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = lineNum;
    copyInParams.blockLen = copyLineLength * sizeof(T);
    copyInParams.srcStride = (srcLineLength - copyLineLength) * sizeof(T);
    copyInParams.dstStride = 0;
    DataCopyPad(dstUb, srcGm, copyInParams, padParams);
}

/*
ub->gm: lineNum rows, dstLineLength is the GM row length, copyLineLength the useful part.
*/
template <typename T>
__aicore__ inline void CopyLineAlignOut(
    const GlobalTensor<T>& dstGm, const LocalTensor<T>& srcUb, int64_t lineNum, int64_t dstLineLength,
    int64_t copyLineLength)
{
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = lineNum;
    copyOutParams.blockLen = copyLineLength * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (dstLineLength - copyLineLength) * sizeof(T);
    DataCopyPad(dstGm, srcUb, copyOutParams);
}

template <typename T_KV>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV>::RmsNormPostWithMul(
    const LocalTensor<float>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& grammaTensor,
    const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;

    if (rSize <= VL_FP32) {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();
        __local_mem__ T_KV* gamma = (__local_mem__ T_KV*)grammaTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T_KV>(x, reg0, pMask, i * stride);
                LoadTensorForDtypeT<T_KV>(gamma, reg1, pMask, 0);
                AscendC::Reg::Mul(reg2, reg0, reg0, pMask);
                ReduceSum(reg2, reg2, pMask);
                AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
                AscendC::Reg::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                AscendC::Reg::Div(reg6, reg0, reg5, pMask);
                AscendC::Reg::Mul(reg7, reg1, reg6, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg7, pMask, i * stride);
            }
        }
    } else {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();
        __local_mem__ T_KV* x_1 = (__local_mem__ T_KV*)xTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ T_KV* gamma = (__local_mem__ T_KV*)grammaTensor.GetPhyAddr();
        __local_mem__ T_KV* gamma_1 = (__local_mem__ T_KV*)grammaTensor.GetPhyAddr() + VL_FP32;
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::Reg::RegTensor<float> reg0, reg1, reg0_1, reg1_1, reg2, reg2_1;
            AscendC::Reg::RegTensor<float> reg3, reg4, reg5, reg6, reg7, reg8, reg9;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeT<T_KV>(x, reg0, pFull, i * stride);
                LoadTensorForDtypeT<T_KV>(x_1, reg0_1, pMask, i * stride);
                LoadTensorForDtypeT<T_KV>(gamma, reg1, pFull, 0);
                LoadTensorForDtypeT<T_KV>(gamma_1, reg1_1, pMask, 0);
                AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
                AscendC::Reg::Mul(reg2_1, reg0_1, reg0_1, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                ReduceSum(reg2, reg2, pFull);

                AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
                AscendC::Reg::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                AscendC::Reg::Div(reg6, reg0, reg5, pFull);
                AscendC::Reg::Mul(reg7, reg1, reg6, pFull);
                StoreTensorForDtypeTOut<float>(dst, reg7, pFull, i * stride);
                AscendC::Reg::Div(reg8, reg0_1, reg5, pMask);
                AscendC::Reg::Mul(reg9, reg1_1, reg8, pMask);
                StoreTensorForDtypeTOut<float>(dst, reg9, pMask, i * stride + VL_FP32);
            }
        }
    }
}

template <typename T_KV>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV>::RmsNormBasicComputeVF(
    const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal, const LocalTensor<float>& wsLocal,
    int64_t aSize, int64_t& foldPoint, int64_t& outerLoopDstStride)
{
    int64_t rSize = dv;
    int64_t stride = dvAlign;
    if (aSize <= 0 || rSize <= 0) {
        return;
    }

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __local_mem__ float* dst = (__local_mem__ float*)wsLocal.GetPhyAddr();

    __local_mem__ T_KV* foldSrcX0A = (__local_mem__ T_KV*)xLocal.GetPhyAddr();
    __local_mem__ T_KV* foldSrcX0B = (__local_mem__ T_KV*)xLocal.GetPhyAddr() + foldSrcBOffset;
    __local_mem__ T_KV* tailSrcX0A = (__local_mem__ T_KV*)xLocal.GetPhyAddr() + tailSrcAOffset;
    __local_mem__ T_KV* tailSrcX0B = (__local_mem__ T_KV*)xLocal.GetPhyAddr() + tailSrcBOffset;
    __local_mem__ T_KV* unFoldX0 = (__local_mem__ T_KV*)xLocal.GetPhyAddr() + unFoldSrcOffset;

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignReg UReg;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__local_mem__ float*)wsLocal.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg2_1;
                LoadTensorForDtypeT<T_KV>(foldSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeT<T_KV>(foldSrcX0B, reg1, pFull, i * outerLoopStride + j * innerLoopStride);

                AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
                AscendC::Reg::Mul(reg2_1, reg1, reg1, pFull);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2, reg2, reg2_1, pFull);
                ReduceSum(reg2, reg2, pFull);
                AscendC::Reg::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg2_1;
                AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadTensorForDtypeT<T_KV>(tailSrcX0A, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeT<T_KV>(tailSrcX0B, reg1, pMask, i * outerLoopStride + j * innerLoopStride);
                AscendC::Reg::Mul(reg2, reg0, reg0, pFull);
                AscendC::Reg::Mul(reg2_1, reg1, reg1, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg2_1, reg2, reg2_1, pMask);
                Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(reg2, reg2_1, pMask);
                ReduceSum(reg2, reg2, pFull);
                AscendC::Reg::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> reg0, reg1;
                LoadTensorForDtypeT<T_KV>(unFoldX0, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                AscendC::Reg::Mul(reg1, reg0, reg0, pFull);
                ReduceSum(reg1, reg1, pFull);
                AscendC::Reg::DataCopyUnAlign((__local_mem__ float*&)dst, reg1, UReg, 1);
            }
            AscendC::Reg::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
}

// Ordinary RMSNorm post-processing (non-quantized).
template <typename T_KV>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV>::RmsNormPostVF(
    const LocalTensor<T_KV>& dstTensor, const LocalTensor<T_KV>& xTensor, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    uint16_t rLoopCount = dvLoopCount;
    uint32_t oriR = static_cast<uint32_t>(dv);
    uint32_t oriRAligned = static_cast<uint32_t>(dvAlign);

    if (rSize <= VL_FP32) {
        __local_mem__ T_KV* dst = (__local_mem__ T_KV*)dstTensor.GetPhyAddr();
        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();
        __local_mem__ T_KV* gamma = (__local_mem__ T_KV*)gammaLocal.GetPhyAddr();
        __local_mem__ float* sumTmp = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(reg0, (__local_mem__ float*)sumTmp + i * stride);
                ReduceSum(reg1, reg0, pMask);
                AscendC::Reg::Muls(reg2, reg1, reciprocal, pFull);
                AscendC::Reg::Adds(reg3, reg2, epsilon, pFull);
                AscendC::Reg::Sqrt(reg4, reg3, pFull);
                Duplicate(reg5, reg4, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                }
            }
        }
    } else {
        __local_mem__ T_KV* dst = (__local_mem__ T_KV*)dstTensor.GetPhyAddr();
        __local_mem__ float* sumTmpA = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
        __local_mem__ float* sumTmpB = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xTensor.GetPhyAddr();
        __local_mem__ T_KV* gamma = (__local_mem__ T_KV*)gammaLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::Reg::RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(reg0, (__local_mem__ float*)sumTmpA + i * stride);
                DataCopy(reg1, (__local_mem__ float*)sumTmpB + i * stride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                ReduceSum(reg2, reg0, pFull);
                AscendC::Reg::Muls(reg3, reg2, reciprocal, pFull);
                AscendC::Reg::Adds(reg4, reg3, epsilon, pFull);
                AscendC::Reg::Sqrt(reg5, reg4, pFull);
                Duplicate(reg5, reg5, pFull);
                uint32_t sreg0 = oriR;
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::Reg::UpdateMask<float>(sreg0);
                    uint32_t addrPtr = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeT<T_KV>(x, reg1, maskOri, addrPtr);
                    AscendC::Reg::Div(reg6, reg1, reg5, maskOri);
                    LoadTensorForDtypeT<T_KV>(gamma, reg7, maskOri, j * VL_FP32);
                    AscendC::Reg::Mul(reg8, reg7, reg6, maskOri);
                    StoreTensorForDtypeTOut<T_KV>(dst, reg8, maskOri, addrPtr);
                }
            }
        }
    }
}

// Ordinary RMSNorm entry (non-quantized).
template <typename T_KV>
__aicore__ inline void KvRmsNormRopeCacheRegbase<T_KV>::RmsNormVF(
    const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
    const LocalTensor<float>& wsLocal, int64_t calcRow)
{
    int64_t stride = dvAlign;
    uint16_t row = static_cast<uint16_t>(calcRow);
    uint32_t actualCol = static_cast<uint32_t>(dv);
    uint32_t alignedCol = static_cast<uint32_t>(dvAlign);
    uint16_t colLoopCount = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(actualCol), VL_FP32));
    if (row <= 0 || actualCol <= 0) {
        return;
    }
    if (actualCol <= CONST_TWO * VL_FP32) {
        RmsNormPostWithMul(wsLocal, xLocal, gammaLocal, row, actualCol, stride);
        __local_mem__ T_KV* dst = (__local_mem__ T_KV*)outVLocal.GetPhyAddr();
        __local_mem__ float* ws = (__local_mem__ float*)wsLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x;
            AscendC::Reg::MaskReg mask;
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = actualCol;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<float>(sreg);
                    LoadTensorForDtypeT<float>(ws, x, mask, i * VL_FP32 + rowId * alignedCol);
                    StoreTensorForDtypeTOut<T_KV>(dst, x, mask, i * VL_FP32 + rowId * alignedCol);
                }
            }
        }
        return;
    }
    int64_t foldPoint = 0;
    int64_t outerLoopDstStride = 0;
    RmsNormBasicComputeVF(xLocal, gammaLocal, wsLocal, calcRow, foldPoint, outerLoopDstStride);
    RmsNormPostVF(outVLocal, xLocal, gammaLocal, wsLocal, row, foldPoint, outerLoopDstStride);
}

} // namespace KvRmsNormRopeCache
#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_BASE_H
