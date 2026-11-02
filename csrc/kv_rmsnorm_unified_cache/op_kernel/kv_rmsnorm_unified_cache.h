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
 * \file kv_rmsnorm_unified_cache.h
 * \brief Ascend950 (arch35 / DAV_3510) RegBase RMSNorm + RoPE fused write into a
 *        unified 576-wide latent cache.
 *
 * Migration provenance
 * --------------------
 * Derived from ops-transformer 9.1.0 commit 6f0d3ec620b338cab4f2d00322ff2e0d141419d2,
 * posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/
 * kv_rms_norm_rope_cache_regbase_full_load.h (class KvRmsNormRopeCacheRegbaseFullLoad).
 *
 * The RegBase math (RmsNormVF / RmsNormBasicComputeVF / RmsNormPostVF and
 * RopeVF / RopeBasicComputeVF, including the VF reduction order, the DINTLV
 * even/odd split and the FP32 intermediate casts) is copied verbatim.  Only the
 * following are changed:
 *   1. kv source row stride is kvStride instead of dk+dv, so a [T,1344] slice
 *      with a 768-element prefix is consumed without a host contiguous copy;
 *   2. cos/sin source row stride is cosStride;
 *   3. cache scatter rows are addressed at slot*cacheDim with the rope half at
 *      +dv, writing the real unified [.., 576] cache in place (the old PA path
 *      wrote two independent dk/dv-wide caches);
 *   4. index values are staged in UB once per tile and validated against
 *      cacheRows; negative / out-of-range slots skip the cache update but still
 *      produce the returned k_rope/c_kv rows (upstream only checked >= 0);
 *   5. quantization, NZ cache modes, block-table modes, the K/V scale/offset
 *      inputs and their UB buffers are removed because this route is limited to
 *      non-quantized BF16/FP16 PA_BNSD (ND) unified cache.
 * No arithmetic used by the retained path was modified.
 */

#ifndef KV_RMSNORM_UNIFIED_CACHE_H
#define KV_RMSNORM_UNIFIED_CACHE_H

#include "kv_rms_norm_rope_cache_regbase_base.h"
#include "kv_rmsnorm_unified_tiling.h"

namespace KvRmsNormRopeCache {

// Upper bound on the number of tokens whose cache rows are coalesced into one
// DataCopyPad. The UB index tile is ubFactor rows, so this only affects the
// scatter burst count, never the address arithmetic.
constexpr int64_t MAX_CACHE_RUN = 64;

template <typename T_KV>
class KvRmsNormUnifiedCache : public KvRmsNormRopeCacheRegbase<T_KV>
{
public:
    __aicore__ inline KvRmsNormUnifiedCache(TPipe* pipe, const KvRmsNormUnifiedTiling* tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR cache, GM_ADDR kOut,
        GM_ADDR vOut)
    {
        int64_t currentBlockFactor = tilingData_->blockFactor;
        if (this->blockIdx == (this->useCoreNum - 1)) {
            currentBlockFactor =
                tilingData_->tokens - (this->useCoreNum - 1) * tilingData_->blockFactor;
        }
        rowsThisCore_ = currentBlockFactor;
        if (rowsThisCore_ <= 0) {
            return;
        }
        this->reciprocal = tilingData_->reciprocal;
        this->epsilon = tilingData_->epsilon;
        this->dv = tilingData_->dv;
        this->dvAlign = tilingData_->dvAlign;
        this->dvB8Align = tilingData_->dvB8Align;
        this->dkAlign = tilingData_->dkAlign;
        this->dk = tilingData_->dk;
        this->ubFactor = tilingData_->ubFactor;
        cacheDim_ = tilingData_->cacheDim;
        cacheRows_ = tilingData_->cacheRows;
        kvStride_ = tilingData_->kvStride;

        // ubLoop: number of UB tiles needed for this core's rows.
        this->ubLoop = (rowsThisCore_ + this->ubFactor - 1) / this->ubFactor;
        this->ubTail = rowsThisCore_ - (this->ubLoop - 1) * this->ubFactor;
        kVFLoop = ops::FloorDiv(tilingData_->halfDk, static_cast<int64_t>(VL_FP32));
        kVFTail = tilingData_->halfDk % VL_FP32;
        this->dvLoopCount = ops::CeilDiv(tilingData_->dv, static_cast<int64_t>(VL_FP32));

        this->kvGm.SetGlobalBuffer(
            (__gm__ T_KV*)kv + static_cast<int64_t>(this->blockIdx) * tilingData_->blockFactor * kvStride_);
        this->gammaGm.SetGlobalBuffer((__gm__ T_KV*)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ T_KV*)cos);
        this->sinGm.SetGlobalBuffer((__gm__ T_KV*)sin);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
        // The unified row is [latent dv | rope dk]. Two sub-views share the row
        // base so both scatters target slot*cacheDim (+dv for rope).
        this->vCacheGm.SetGlobalBuffer((__gm__ T_KV*)cache);
        this->kCacheGm.SetGlobalBuffer((__gm__ T_KV*)cache + tilingData_->dv);
        this->kOutGm.SetGlobalBuffer(
            (__gm__ T_KV*)kOut + static_cast<int64_t>(this->blockIdx) * tilingData_->blockFactor * tilingData_->dk);
        this->vOutGm.SetGlobalBuffer(
            (__gm__ T_KV*)vOut + static_cast<int64_t>(this->blockIdx) * tilingData_->blockFactor * tilingData_->dv);

        pipe_->InitBuffer(inQueueGamma, BUFFER_COUNT_SINGLE, tilingData_->dvAlign * sizeof(T_KV));
        pipe_->InitBuffer(
            inQueueCosSin, BUFFER_COUNT_DOUBLE,
            this->ubFactor * COS_SIN_CHANNEL_COUNT * tilingData_->halfDkAlign * sizeof(T_KV));
        pipe_->InitBuffer(inQueueX, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->inUbSize);
        pipe_->InitBuffer(outQueue, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->outUbSize);
        pipe_->InitBuffer(wsBuffer0, WORKSPACE_FLOAT_VECTOR_COUNT * VL_FP32 * sizeof(float));
        pipe_->InitBuffer(wsBuffer1, this->ubFactor * tilingData_->rmsNormWspSize);
        // int64 slot ids for one UB tile, 32-byte aligned.
        const int64_t indexBufBytes =
            (this->ubFactor * static_cast<int64_t>(sizeof(int64_t)) + 31) / 32 * 32;
        pipe_->InitBuffer(indexBuf, indexBufBytes);
    }

    __aicore__ inline void RopeBasicComputeVF(
        __local_mem__ T_KV* x, AscendC::Reg::RegTensor<float>& x_0, AscendC::Reg::RegTensor<float>& x_1,
        AscendC::Reg::RegTensor<float>& cos_0, AscendC::Reg::RegTensor<float>& cos_1,
        AscendC::Reg::RegTensor<float>& sin_0, AscendC::Reg::RegTensor<float>& sin_1,
        __local_mem__ T_KV* realCos, __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin,
        __local_mem__ T_KV* imgSin, __local_mem__ float* ws, uint32_t xOffset, uint32_t cosSinOffset, uint16_t rowId,
        uint16_t colId, AscendC::Reg::MaskReg mask, AscendC::Reg::MaskReg maskAll)
    {
        // cast to float32, then de-interleave even/odd lanes (verbatim upstream)
        LoadTensorForDtypeT<T_KV>(x, x_0, maskAll, colId * CONST_TWO * VL_FP32 + rowId * xOffset);
        LoadTensorForDtypeT<T_KV>(x, x_1, maskAll, (colId * CONST_TWO + CONST_ONE) * VL_FP32 + rowId * xOffset);
        StoreTensorForDtypeTOut<float>(ws, x_0, maskAll, 0);
        StoreTensorForDtypeTOut<float>(ws, x_1, maskAll, VL_FP32);
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        DataCopy<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(x_0, x_1, ((__local_mem__ float*)(ws)));
        LoadTensorForDtypeT(realCos, cos_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgCos, cos_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(realSin, sin_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgSin, sin_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        AscendC::Reg::Mul(cos_0, x_0, cos_0, mask);
        AscendC::Reg::Mul(cos_1, x_1, cos_1, mask);
        AscendC::Reg::Muls(x_1, x_1, -1, mask);
        AscendC::Reg::Mul(sin_0, x_1, sin_0, mask);
        AscendC::Reg::Mul(sin_1, x_0, sin_1, mask);
        AscendC::Reg::Add(sin_0, cos_0, sin_0, mask);
        AscendC::Reg::Add(sin_1, cos_1, sin_1, mask);
    }

    __aicore__ inline void RopeVF(
        __local_mem__ T_KV* outFront, __local_mem__ T_KV* outBack, __local_mem__ T_KV* x, __local_mem__ T_KV* realCos,
        __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin, __local_mem__ T_KV* imgSin, __local_mem__ float* ws,
        uint16_t row, uint16_t colLoopCount, uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1;
            AscendC::Reg::UnalignReg UReg0;
            AscendC::Reg::UnalignReg UReg1;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg maskAll =
                AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __local_mem__ T_KV* out0 = outFront + rowId * xOffset;
                __local_mem__ T_KV* out1 = outBack + rowId * xOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                }
                mask = AscendC::Reg::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, kVFTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, kVFTail);
                AscendC::Reg::DataCopyUnAlignPost(out0, UReg0, 0);
                AscendC::Reg::DataCopyUnAlignPost(out1, UReg1, 0);
            }
        }
    }

    __aicore__ inline void Rope(
        const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& realCosLocal, const LocalTensor<T_KV>& imgCosLocal,
        const LocalTensor<T_KV>& realSinLocal, const LocalTensor<T_KV>& imgSinLocal,
        const LocalTensor<float>& wsLocal, uint16_t rowNum)
    {
        uint32_t colCosSin = static_cast<uint32_t>(tilingData_->halfDk);
        uint32_t xOffset = static_cast<uint32_t>(tilingData_->dkAlign);
        uint32_t cosSinOffset = static_cast<uint32_t>(tilingData_->halfDkAlign);

        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xLocal.GetPhyAddr();
        __local_mem__ T_KV* realCos = (__local_mem__ T_KV*)realCosLocal.GetPhyAddr();
        __local_mem__ T_KV* imgCos = (__local_mem__ T_KV*)imgCosLocal.GetPhyAddr();
        __local_mem__ T_KV* realSin = (__local_mem__ T_KV*)realSinLocal.GetPhyAddr();
        __local_mem__ T_KV* imgSin = (__local_mem__ T_KV*)imgSinLocal.GetPhyAddr();
        __local_mem__ float* ws = (__local_mem__ float*)wsLocal.GetPhyAddr();

        kOutLocal_ = outQueue.AllocTensor<T_KV>();
        __local_mem__ T_KV* outFront = (__local_mem__ T_KV*)kOutLocal_.GetPhyAddr();
        __local_mem__ T_KV* outBack = (__local_mem__ T_KV*)kOutLocal_.GetPhyAddr() + tilingData_->halfDk;
        RopeVF(
            outFront, outBack, x, realCos, imgCos, realSin, imgSin, ws, rowNum, kVFLoop, colCosSin, xOffset,
            cosSinOffset);
        outQueue.EnQue(kOutLocal_);
        kOutLocal_ = outQueue.DeQue<T_KV>();
    }

    __aicore__ inline void RmsNorm(
        const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
        const LocalTensor<float>& wsLocal, int64_t calcRow)
    {
        this->RmsNormVF(outVLocal, xLocal, gammaLocal, wsLocal, calcRow);
    }

    // Scatter one UB tile of already computed rows into the unified cache.
    // `cacheFront` points at the latent half (base) or rope half (base + dv).
    // The caller has already staged slots for this tile into indexLocal.
    __aicore__ inline void ScatterUnified(
        const GlobalTensor<T_KV>& cacheGm, const LocalTensor<T_KV>& srcLocal, const LocalTensor<int64_t>& indexLocal,
        int64_t rows, int64_t rowStride, int64_t copyLen)
    {
        int64_t i = 0;
        while (i < rows) {
            int64_t slot = indexLocal.GetValue(i);
            if (slot < 0 || slot >= cacheRows_) {
                ++i;
                continue;
            }
            int64_t run = 1;
            while (run < MAX_CACHE_RUN && i + run < rows) {
                int64_t next = indexLocal.GetValue(i + run);
                if (next != slot + run || next >= cacheRows_) {
                    break;
                }
                ++run;
            }
            DataCopyExtParams params;
            params.blockCount = static_cast<uint16_t>(run);
            params.blockLen = static_cast<uint32_t>(copyLen * sizeof(T_KV));
            params.srcStride = 0; // UB rows are packed at rowStride == copyLen
            params.dstStride = static_cast<uint32_t>((cacheDim_ - copyLen) * sizeof(T_KV));
            // The MTE3 destination base is derived from a scalar slot value.
            // MTE2->S was issued once when the slot tile was staged; this
            // S->MTE3 pair additionally orders the scalar read against the DMA
            // that consumes it, mirroring upstream ScatterUpdateK/V's explicit
            // SetFlag/WaitFlag<HardEvent::S_MTE3> around index-derived copies.
            event_t eventS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventS);
            WaitFlag<HardEvent::S_MTE3>(eventS);
            DataCopyPad(cacheGm[slot * cacheDim_], srcLocal[i * rowStride], params);
            i += run;
        }
    }

    __aicore__ inline void Process()
    {
        if (rowsThisCore_ <= 0) {
            return;
        }
        LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
        CopyInLineAlign<T_KV>(gammaLocal, this->gammaGm, 1, tilingData_->dv, tilingData_->dv);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<T_KV>();
        LocalTensor<float> workspaceBuffer0 = wsBuffer0.Get<float>();
        LocalTensor<float> workspaceBuffer1 = wsBuffer1.Get<float>();

        int64_t calcRow = this->ubFactor;
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            if (loopIdx == this->ubLoop - 1) {
                calcRow = this->ubTail;
            }
            int64_t startIdx = static_cast<int64_t>(this->blockIdx) * tilingData_->blockFactor +
                               loopIdx * this->ubFactor;
            int64_t kvRowOffset = loopIdx * this->ubFactor * kvStride_;
            int64_t cosRowOffset = startIdx * tilingData_->cosStride;
            int64_t kOutGmOffset = loopIdx * this->ubFactor * tilingData_->dk;
            int64_t vOutGmOffset = loopIdx * this->ubFactor * tilingData_->dv;

            // Stage slot ids once per tile (scalar GM reads on the hot path are avoided).
            LocalTensor<int64_t> indexLocal = indexBuf.Get<int64_t>();
            DataCopyExtParams idxParams;
            idxParams.blockCount = 1;
            idxParams.blockLen = static_cast<uint32_t>(calcRow * sizeof(int64_t));
            idxParams.srcStride = 0;
            idxParams.dstStride = 0;
            DataCopyPadExtParams<int64_t> idxPad{false, 0, 0, 0};
            DataCopyPad(indexLocal, this->indexGm[startIdx], idxParams, idxPad);
            event_t eventIdx = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIdx);
            WaitFlag<HardEvent::MTE2_S>(eventIdx);

            // RoPE input: last dk columns of each kv row.
            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> realCosLocal = inQueueCosSin.AllocTensor<T_KV>();
            LocalTensor<T_KV> imgCosLocal = realCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_KV> realSinLocal = imgCosLocal[this->ubFactor * tilingData_->halfDkAlign];
            LocalTensor<T_KV> imgSinLocal = realSinLocal[this->ubFactor * tilingData_->halfDkAlign];
            CopyInLineAlign<T_KV>(
                ropeLocal, this->kvGm[kvRowOffset + tilingData_->dv], calcRow, kvStride_, tilingData_->dk);
            CopyInLineAlign<T_KV>(
                realCosLocal, this->cosGm[cosRowOffset], calcRow, tilingData_->cosStride, tilingData_->halfDk);
            CopyInLineAlign<T_KV>(
                imgCosLocal, this->cosGm[cosRowOffset + tilingData_->halfDk], calcRow, tilingData_->cosStride,
                tilingData_->halfDk);
            CopyInLineAlign<T_KV>(
                realSinLocal, this->sinGm[cosRowOffset], calcRow, tilingData_->cosStride, tilingData_->halfDk);
            CopyInLineAlign<T_KV>(
                imgSinLocal, this->sinGm[cosRowOffset + tilingData_->halfDk], calcRow, tilingData_->cosStride,
                tilingData_->halfDk);
            inQueueX.EnQue(ropeLocal);
            inQueueCosSin.EnQue(realCosLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            realCosLocal = inQueueCosSin.DeQue<T_KV>();

            Rope(ropeLocal, realCosLocal, imgCosLocal, realSinLocal, imgSinLocal, workspaceBuffer0, calcRow);
            inQueueX.FreeTensor(ropeLocal);
            inQueueCosSin.FreeTensor(realCosLocal);
            ScatterUnified(
                this->kCacheGm, kOutLocal_, indexLocal, calcRow, tilingData_->dkAlign, tilingData_->dk);
            CopyLineAlignOut<T_KV>(this->kOutGm[kOutGmOffset], kOutLocal_, calcRow, tilingData_->dk, tilingData_->dk);
            outQueue.FreeTensor(kOutLocal_);

            // RMSNorm input: first dv columns of each kv row.
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            CopyInLineAlign<T_KV>(
                xLocal, this->kvGm[kvRowOffset], calcRow, kvStride_, tilingData_->dv);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();

            vOutLocal_ = outQueue.AllocTensor<T_KV>();
            this->RmsNorm(vOutLocal_, xLocal, gammaLocal, workspaceBuffer1, calcRow);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(vOutLocal_);
            vOutLocal_ = outQueue.DeQue<T_KV>();
            ScatterUnified(
                this->vCacheGm, vOutLocal_, indexLocal, calcRow, tilingData_->dvAlign, tilingData_->dv);
            CopyLineAlignOut<T_KV>(this->vOutGm[vOutGmOffset], vOutLocal_, calcRow, tilingData_->dv, tilingData_->dv);
            outQueue.FreeTensor(vOutLocal_);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    const KvRmsNormUnifiedTiling* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueCosSin;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer0, wsBuffer1;
    TBuf<TPosition::VECCALC> indexBuf;

    LocalTensor<T_KV> kOutLocal_;
    LocalTensor<T_KV> vOutLocal_;

    uint16_t kVFLoop = 0;
    uint32_t kVFTail = 0;
    int64_t cacheDim_ = 576;
    int64_t cacheRows_ = 0;
    int64_t kvStride_ = 576;
    int64_t rowsThisCore_ = 0;
};

} // namespace KvRmsNormRopeCache

#endif // KV_RMSNORM_UNIFIED_CACHE_H
