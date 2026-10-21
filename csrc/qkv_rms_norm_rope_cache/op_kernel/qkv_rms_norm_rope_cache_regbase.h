// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Ascend 950 (DAV_3510) RegBase implementation of qkv_rms_norm_rope_cache.
//
// Modelled after ops-transformer/posembedding/kv_rms_norm_rope_cache/op_kernel/arch35
// ("RegBase" kernels): norm + RoPE run on vector registers inside one __VEC_SCOPE__ per
// UB tile, so each head costs exactly one UB read and one UB write.
//
// Work distribution: three core groups.  Q cores run norm+RoPE into q_out, K cores run
// norm+RoPE and scatter into k_cache, V cores are a pure identity scatter into v_cache.
//
// See OPTIMIZATION.md for the measurements behind each choice.

#ifndef QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
#define QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_

#include "kernel_operator.h"

#include "../op_host/qkv_rms_norm_rope_cache_tiling.h"

namespace sglang {
namespace npu_kernel {

using namespace AscendC;
using namespace AscendC::Reg;

constexpr int64_t kRbDim = 128;
constexpr int64_t kRbVl = 64;  // fp32 lanes per vector register (256B / 4)
constexpr int64_t kRbD0 = 16;  // PA_NZ innermost block (32B for bf16)
constexpr int64_t kRbD1 = kRbDim / kRbD0;
constexpr int64_t kRbMaxTok = 64;  // max tokens a UB tile may span (bounds the cos/sin staging)

constexpr static CastTrait kRbCastB16ToB32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                              RoundMode::UNKNOWN};
constexpr static CastTrait kRbCastFp32ToFp16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                RoundMode::CAST_RINT};

template <HardEvent event>
__aicore__ inline void RbSetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline void RbLoadBf16(__local_mem__ bfloat16_t *p, RegTensor<float> &dst, MaskReg &mask, uint32_t off)
{
    RegTensor<bfloat16_t> r;
    DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(r, p + off);
    Cast<float, bfloat16_t, kRbCastB16ToB32>(dst, r, mask);
}

__aicore__ inline void RbStoreBf16(__local_mem__ bfloat16_t *p, RegTensor<float> &src, MaskReg &mask, uint32_t off)
{
    RegTensor<bfloat16_t> r;
    Cast<bfloat16_t, float, kRbCastFp32ToFp16>(r, src, mask);
    DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(p + off, r, mask);
}

// RMSNorm(gamma) + half-and-half RoPE for a whole UB tile in one vector scope.
// gamma is folded into the RoPE factors once per token (the rotate-half term pairs y_hi with
// sin_lo and y_lo with sin_hi, so each sin factor carries the *other* half's gamma).
__aicore__ inline void RbNormRopeTile(__local_mem__ bfloat16_t *x, __local_mem__ bfloat16_t *out,
                                      __local_mem__ bfloat16_t *gamma, __local_mem__ bfloat16_t *cosU,
                                      __local_mem__ bfloat16_t *sinU, uint16_t rows, int64_t numHeads,
                                      int64_t startIdx, float invDim, float eps)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x0, x1, g0, g1, c0, c1, s0, s1, y0, y1, t0, t1, r;
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();

        RbLoadBf16(gamma, g0, pFull, 0);
        RbLoadBf16(gamma, g1, pFull, static_cast<uint32_t>(kRbVl));

        uint16_t i = 0;
        int64_t nIdx = startIdx % numHeads;
        int64_t tokIdx = 0;
        while (i < rows) {
            const uint16_t left = static_cast<uint16_t>(numHeads - nIdx);
            const uint16_t cnt = (rows - i < left) ? (rows - i) : left;
            const uint32_t facOff = static_cast<uint32_t>(tokIdx * kRbDim);
            RbLoadBf16(cosU, c0, pFull, facOff);
            RbLoadBf16(cosU, c1, pFull, facOff + static_cast<uint32_t>(kRbVl));
            RbLoadBf16(sinU, s0, pFull, facOff);
            RbLoadBf16(sinU, s1, pFull, facOff + static_cast<uint32_t>(kRbVl));
            Mul(c0, c0, g0, pFull);  // g_lo * cos_lo
            Mul(c1, c1, g1, pFull);  // g_hi * cos_hi
            Mul(s0, s0, g1, pFull);  // g_hi * sin_lo
            Mul(s1, s1, g0, pFull);  // g_lo * sin_hi

            for (uint16_t j = 0; j < cnt; ++j) {
                const uint32_t off = static_cast<uint32_t>((i + j) * kRbDim);
                RbLoadBf16(x, x0, pFull, off);
                RbLoadBf16(x, x1, pFull, off + static_cast<uint32_t>(kRbVl));

                Mul(t0, x0, x0, pFull);
                Mul(t1, x1, x1, pFull);
                Add(t0, t0, t1, pFull);
                ReduceSum(r, t0, pFull);
                Muls(r, r, invDim, pFull);
                Adds(r, r, eps, pFull);
                Sqrt(r, r, pFull);
                Duplicate(r, r, pFull);

                Div(y0, x0, r, pFull);
                Div(y1, x1, r, pFull);

                Mul(t0, y1, s0, pFull);
                Mul(t1, y0, s1, pFull);
                Mul(y0, y0, c0, pFull);
                Mul(y1, y1, c1, pFull);
                Sub(y0, y0, t0, pFull);
                Add(y1, y1, t1, pFull);

                RbStoreBf16(out, y0, pFull, off);
                RbStoreBf16(out, y1, pFull, off + static_cast<uint32_t>(kRbVl));
            }
            i = static_cast<uint16_t>(i + cnt);
            nIdx = 0;
            ++tokIdx;
        }
    }
}

class KernelQkvRmsNormRopeCacheRegbase {
public:
    __aicore__ inline KernelQkvRmsNormRopeCacheRegbase(TPipe *pipe, const QkvRmsNormRopeCacheTilingData *tiling)
        : pipe_(pipe), t_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos, GM_ADDR sin,
                                GM_ADDR index, GM_ADDR q_out, GM_ADDR k_cache, GM_ADDR v_cache)
    {
        blockIdx_ = GetBlockIdx();
        nq_ = t_->numHeadQ;
        nk_ = t_->numHeadK;
        nv_ = t_->numHeadV;
        numHead_ = t_->numHead;
        eps_ = t_->epsilon;
        invDim_ = 1.0f / static_cast<float>(t_->qkvDim);
        blockSize_ = t_->blockSize;

        isK_ = blockIdx_ < t_->blockDimK ? 1 : 0;
        isV_ = (blockIdx_ >= t_->blockDimK && blockIdx_ < t_->blockDimK + t_->blockDimV) ? 1 : 0;
        const int64_t startK = blockIdx_ * t_->blockFactorK;
        const int64_t startV = (blockIdx_ - t_->blockDimK) * t_->blockFactorV;
        const int64_t startQ = (blockIdx_ - t_->blockDimK - t_->blockDimV) * t_->blockFactorQ;
        coreStart_ = isK_ ? startK : (isV_ ? startV : startQ);
        coreRows_ = isK_ ? t_->blockFactorK : (isV_ ? t_->blockFactorV : t_->blockFactorQ);
        numHeads_ = isK_ ? nk_ : (isV_ ? nv_ : nq_);
        const int64_t limit = t_->batchSize * t_->seqLength * numHeads_;
        if (coreStart_ + coreRows_ > limit) {
            coreRows_ = limit - coreStart_;
        }
        const bool gammaIsQ = (isK_ == 0 && isV_ == 0);

        qkvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)qkv);
        gammaQGm_.SetGlobalBuffer((__gm__ bfloat16_t *)q_gamma);
        gammaKGm_.SetGlobalBuffer((__gm__ bfloat16_t *)k_gamma);
        cosGm_.SetGlobalBuffer((__gm__ bfloat16_t *)cos);
        sinGm_.SetGlobalBuffer((__gm__ bfloat16_t *)sin);
        indexGm_.SetGlobalBuffer((__gm__ int64_t *)index);
        qOutGm_.SetGlobalBuffer((__gm__ bfloat16_t *)q_out);
        kCacheGm_.SetGlobalBuffer((__gm__ bfloat16_t *)k_cache);
        vCacheGm_.SetGlobalBuffer((__gm__ bfloat16_t *)v_cache);

        ubRows_ = static_cast<int64_t>(t_->ubFactor);
        if (ubRows_ <= 0 || ubRows_ > 512) {
            ubRows_ = 128;
        }
        if (ubRows_ > kRbMaxTok * numHeads_) {
            ubRows_ = kRbMaxTok * numHeads_;
        }
        const int64_t tokMax = ubRows_ / numHeads_ + 2;

        pipe_->InitBuffer(inQueue_, 2, ubRows_ * kRbDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, 2, ubRows_ * kRbDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(gammaBuf_, kRbDim * sizeof(bfloat16_t));
        pipe_->InitBuffer(cosSinBuf_, tokMax * 2 * kRbDim * sizeof(bfloat16_t));

        LocalTensor<bfloat16_t> g = gammaBuf_.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> &src = gammaIsQ ? gammaQGm_ : gammaKGm_;
        DataCopyExtParams gp{1, static_cast<uint32_t>(kRbDim * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyPad(g, src, gp, pad);
        RbSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void Process()
    {
        if (coreRows_ <= 0) {
            return;
        }
        const int64_t headBase = isK_ ? nq_ : (isV_ ? (nq_ + nk_) : 0);
        ProcessHeads(coreStart_, coreStart_ + coreRows_, numHeads_, headBase, isK_ != 0);
    }

private:
    __aicore__ inline void ProcessHeads(int64_t startHead, int64_t endHead, int64_t numHeads, int64_t headBase,
                                        bool isK)
    {
        const int64_t limit = t_->batchSize * t_->seqLength * numHeads;
        if (startHead >= limit) {
            return;
        }
        const int64_t total = ((endHead < limit) ? endHead : limit) - startHead;
        const bool isVsection = (!isK) && (headBase != 0);

        for (int64_t done = 0; done < total; done += ubRows_) {
            if (done > 0) {
                // WAR on the output queue: the queue only tracks V, so make sure the previous
                // tile's MTE3 (q_out store / PA_NZ scatter) finished reading its buffer.
                RbSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
            const int64_t rows = (total - done < ubRows_) ? (total - done) : ubRows_;
            const int64_t start = startHead + done;
            const int64_t token0 = start / numHeads;
            const int64_t token1 = (start + rows - 1) / numHeads;
            const int64_t ntok = token1 - token0 + 1;

            LocalTensor<bfloat16_t> xL = inQueue_.AllocTensor<bfloat16_t>();
            LoadQkv(xL, rows, start, numHeads, headBase);
            inQueue_.EnQue(xL);
            xL = inQueue_.DeQue<bfloat16_t>();

            if (isVsection) {
                // V is an identity copy: scatter straight from the input staging buffer.
                RbSetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                Scatter(xL, rows, start, numHeads, false);
                RbSetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                inQueue_.FreeTensor(xL);
                continue;
            }

            LocalTensor<bfloat16_t> oL = outQueue_.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> facL = cosSinBuf_.Get<bfloat16_t>();
            // WAR: the previous tile's VF may still be reading the cos/sin buffer.
            RbSetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            LoadRopeFac(facL, token0, ntok);
            RbSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            RbNormRopeTile((__local_mem__ bfloat16_t *)xL.GetPhyAddr(), (__local_mem__ bfloat16_t *)oL.GetPhyAddr(),
                           (__local_mem__ bfloat16_t *)gammaBuf_.Get<bfloat16_t>().GetPhyAddr(),
                           (__local_mem__ bfloat16_t *)facL.GetPhyAddr(),
                           (__local_mem__ bfloat16_t *)facL.GetPhyAddr() + ntok * kRbDim,
                           static_cast<uint16_t>(rows), numHeads, start, invDim_, eps_);
            outQueue_.EnQue(oL);
            oL = outQueue_.DeQue<bfloat16_t>();

            if (isK) {
                Scatter(oL, rows, start, numHeads, true);
            } else {
                DataCopyExtParams p{1, static_cast<uint32_t>(rows * kRbDim * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(qOutGm_[start * kRbDim], oL, p);
            }
            outQueue_.FreeTensor(oL);
            inQueue_.FreeTensor(xL);
        }
    }

    __aicore__ inline void LoadQkv(const LocalTensor<bfloat16_t> &dst, int64_t rows, int64_t start,
                                   int64_t numHeads, int64_t headBase)
    {
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyExtParams cp{1, 0, 0, 0, 0};
        int64_t done = 0;
        while (done < rows) {
            const int64_t g = start + done;
            const int64_t token = g / numHeads;
            const int64_t nIdx = g % numHeads;
            const int64_t cnt = (rows - done < numHeads - nIdx) ? (rows - done) : (numHeads - nIdx);
            const int64_t off = token * (numHead_ * kRbDim) + headBase * kRbDim + nIdx * kRbDim;
            cp.blockLen = static_cast<uint32_t>(cnt * kRbDim * sizeof(bfloat16_t));
            DataCopyPad(dst[done * kRbDim], qkvGm_[off], cp, pad);
            done += cnt;
        }
    }

    __aicore__ inline void LoadRopeFac(const LocalTensor<bfloat16_t> &fac, int64_t token0, int64_t ntok)
    {
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyExtParams cp{1, static_cast<uint32_t>(ntok * kRbDim * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(fac, cosGm_[token0 * kRbDim], cp, pad);
        DataCopyPad(fac[ntok * kRbDim], sinGm_[token0 * kRbDim], cp, pad);
    }

    // One head of one token: its D/16 slices are blockSize*32B apart in the PA_NZ cache.
    __aicore__ inline void ScatterOneHead(GlobalTensor<bfloat16_t> &cache, const LocalTensor<bfloat16_t> &src,
                                          int64_t srcRow, int64_t headIdx, int64_t page, int64_t off,
                                          int64_t numHeads)
    {
        DataCopyExtParams cp{static_cast<uint16_t>(kRbD1), static_cast<uint32_t>(kRbD0 * sizeof(bfloat16_t)), 0,
                             static_cast<uint32_t>(blockSize_ * kRbD0 * sizeof(bfloat16_t)) -
                                 static_cast<uint32_t>(kRbD0 * sizeof(bfloat16_t)),
                             0};
        const int64_t base = page * (numHeads * kRbD1 * blockSize_ * kRbD0) +
                             headIdx * (kRbD1 * blockSize_ * kRbD0) + off * kRbD0;
        DataCopyPad(cache[base], src[srcRow * kRbDim], cp);
    }

    // Consecutive slots inside one page: n tokens of one (head, D/16) block are contiguous in
    // the cache, so they go out as one n*32B burst.  DataCopy works in 32B blocks and every
    // address here is 32B aligned.
    __aicore__ inline void ScatterRun(GlobalTensor<bfloat16_t> &cache, const LocalTensor<bfloat16_t> &src,
                                      int64_t firstRow, int64_t ntok, int64_t slot, int64_t numHeads)
    {
        const int64_t page = slot / blockSize_;
        const int64_t off = slot % blockSize_;
        DataCopyParams cp;
        cp.blockCount = static_cast<uint16_t>(ntok);
        cp.blockLen = 1;  // 32B
        cp.srcStride = static_cast<uint16_t>(numHeads * kRbDim / kRbD0 - 1);
        cp.dstStride = 0;
        for (int64_t h = 0; h < numHeads; ++h) {
            const int64_t srcHead = (firstRow + h) * kRbDim;
            for (int64_t d1 = 0; d1 < kRbD1; ++d1) {
                const int64_t dst = page * (numHeads * kRbD1 * blockSize_ * kRbD0) +
                                    h * (kRbD1 * blockSize_ * kRbD0) + d1 * (blockSize_ * kRbD0) + off * kRbD0;
                DataCopy(cache[dst], src[srcHead + d1 * kRbD0], cp);
            }
        }
    }

    __aicore__ inline void Scatter(const LocalTensor<bfloat16_t> &src, int64_t rows, int64_t start,
                                   int64_t numHeads, bool isK)
    {
        GlobalTensor<bfloat16_t> &cache = isK ? kCacheGm_ : vCacheGm_;
        int64_t pos = 0;
        while (pos < rows) {
            const int64_t g = start + pos;
            const int64_t token = g / numHeads;
            const int64_t nIdx = g % numHeads;
            const int64_t cnt = (rows - pos < numHeads - nIdx) ? (rows - pos) : (numHeads - nIdx);
            if (nIdx != 0 || cnt != numHeads) {
                // partial token at a tile boundary: per-head stores
                const int64_t slot = indexGm_(token);
                if (slot >= 0) {
                    const int64_t page = slot / blockSize_;
                    const int64_t off = slot % blockSize_;
                    for (int64_t j = 0; j < cnt; ++j) {
                        ScatterOneHead(cache, src, pos + j, nIdx + j, page, off, numHeads);
                    }
                }
                pos += cnt;
                continue;
            }
            const int64_t ntok = (rows - pos) / numHeads;
            int64_t t = 0;
            while (t < ntok) {
                const int64_t slot = indexGm_(token + t);
                if (slot < 0) {
                    ++t;
                    continue;
                }
                const int64_t off = slot % blockSize_;
                int64_t n = 1;
                while (t + n < ntok && off + n < blockSize_ && indexGm_(token + t + n) == slot + n) {
                    ++n;
                }
                if (n == 1) {
                    const int64_t page = slot / blockSize_;
                    for (int64_t h = 0; h < numHeads; ++h) {
                        ScatterOneHead(cache, src, pos + t * numHeads + h, h, page, off, numHeads);
                    }
                } else {
                    ScatterRun(cache, src, pos + t * numHeads, n, slot, numHeads);
                }
                t += n;
            }
            pos += ntok * numHeads;
        }
    }

    TPipe *pipe_;
    const QkvRmsNormRopeCacheTilingData *t_;
    int64_t blockIdx_ = 0;
    int64_t nq_ = 0, nk_ = 0, nv_ = 0, numHead_ = 0;
    int64_t numHeads_ = 0;
    int64_t blockSize_ = 0;
    int64_t coreStart_ = 0, coreRows_ = 0;
    int64_t ubRows_ = 0;
    int64_t isK_ = 0, isV_ = 0;
    float eps_ = 1e-6f;
    float invDim_ = 1.0f / 128.0f;

    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> gammaBuf_;
    TBuf<TPosition::VECCALC> cosSinBuf_;

    GlobalTensor<bfloat16_t> qkvGm_, gammaQGm_, gammaKGm_, cosGm_, sinGm_;
    GlobalTensor<bfloat16_t> qOutGm_, kCacheGm_, vCacheGm_;
    GlobalTensor<int64_t> indexGm_;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // QKV_RMS_NORM_ROPE_CACHE_REGBASE_H_
