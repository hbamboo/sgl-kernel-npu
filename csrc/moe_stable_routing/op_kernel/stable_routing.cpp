// Isolated E64/K4/H2048 raw-byte routing proof. No production integration.
#include "kernel_operator.h"

using namespace AscendC;
namespace {
constexpr uint32_t E = 64, K = 4, H = 2048, S = 64;
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
template<HardEvent event> __aicore__ inline void Edge(TPipe& pipe) {
    auto id = static_cast<event_t>(pipe.FetchEventID(event));
    SetFlag<event>(id); WaitFlag<event>(id);
}

class LocalRanks {
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ids_, key_, seq_, idx_, concat_, sorted_, temp_;
    TBuf<TPosition::VECCALC> skey_, sidx_, expertOff_, gathered_, ranks_, pos_, starts_, counts_;
    GlobalTensor<int32_t> idsGm_, rankGm_, countGm_;
    uint32_t begin_, count_, n_, stride_;
public:
    __aicore__ inline void Init(GM_ADDR ids, GM_ADDR ranks, GM_ADDR counts,
                                uint32_t tokens, uint32_t perCore, uint32_t n, uint32_t stride) {
        begin_ = GetBlockIdx() * perCore * K;
        count_ = Min(perCore, tokens - GetBlockIdx() * perCore) * K;
        n_ = n; stride_ = stride;
        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        rankGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
        countGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts));
        pipe_.InitBuffer(ids_, n*4); pipe_.InitBuffer(key_, n*4); pipe_.InitBuffer(seq_, n*4);
        pipe_.InitBuffer(idx_, n*4); pipe_.InitBuffer(concat_, n*8);
        pipe_.InitBuffer(sorted_, n*8); pipe_.InitBuffer(temp_, n*8);
        pipe_.InitBuffer(skey_, n*4); pipe_.InitBuffer(sidx_, n*4);
        pipe_.InitBuffer(expertOff_, n*4); pipe_.InitBuffer(gathered_, n*4);
        pipe_.InitBuffer(ranks_, n*4); pipe_.InitBuffer(pos_, n*4);
        // Entry64 is for padded expert64, never part of the public histogram.
        pipe_.InitBuffer(starts_, 288); pipe_.InitBuffer(counts_, 256);
    }
    __aicore__ inline void Run() {
        auto ids = ids_.Get<int32_t>(); auto key = key_.Get<float>();
        auto seq = seq_.Get<float>(); auto idx = idx_.Get<uint32_t>();
        Duplicate(ids, int32_t(E), n_);
        Edge<HardEvent::V_MTE2>(pipe_);
        DataCopyExtParams cp{1, count_*4, 0, 0, 0};
        // MTE2 rounds the final UB block to 32B: explicitly retain expert64
        // in that block; the rest of the Sort padding was initialized above.
        DataCopyPadExtParams<int32_t> pad{true, 0, static_cast<uint8_t>((8-count_%8)%8), int32_t(E)};
        DataCopyPad(ids, idsGm_[begin_], cp, pad);
        Edge<HardEvent::MTE2_V>(pipe_);
        Cast(key, ids, RoundMode::CAST_NONE, n_);
        Muls(key, key, static_cast<float>(stride_), n_);
        ArithProgression(seq, static_cast<float>(begin_), 1.0f, n_);
        Add(key, key, seq, n_); Muls(key, key, -1.0f, n_);
        ArithProgression(idx.ReinterpretCast<int32_t>(), int32_t(0), int32_t(1), n_);
        LocalTensor<float> proposal;
        auto concat = concat_.Get<float>(); auto sorted = sorted_.Get<float>(); auto tmp = temp_.Get<float>();
        Concat(proposal, key, concat, n_/32);
        Sort<float, true>(sorted, proposal, idx, tmp, n_/32);
        auto skey = skey_.Get<float>(); auto sidx = sidx_.Get<uint32_t>();
        Extract(skey, sidx, sorted, n_/32);
        Edge<HardEvent::V_S>(pipe_);
        auto starts = starts_.Get<int32_t>(); auto counts = counts_.Get<int32_t>();
        // Only fixed expert metadata is scalar: 64 lower_bounds, not a scalar slot loop.
        uint32_t previous = 0;
        starts.SetValue(0, 0);
        for (uint32_t e = 1; e <= E; ++e) {
            uint32_t lo = previous, hi = count_;
            float threshold = -static_cast<float>(e * stride_);
            while (lo < hi) {
                uint32_t mid = (lo + hi) / 2;
                if (skey.GetValue(mid) > threshold) lo = mid + 1;
                else hi = mid;
            }
            starts.SetValue(e, static_cast<int32_t>(lo));
            counts.SetValue(e-1, static_cast<int32_t>(lo-previous));
            previous = lo;
        }
        Edge<HardEvent::S_V>(pipe_);
        // Recover expert from exact power-of-two encoded keys, gather group starts,
        // and invert the sorted source permutation with vector Scatter (byte offsets).
        auto expert = expertOff_.Get<int32_t>(); auto gathered = gathered_.Get<int32_t>();
        auto rank = ranks_.Get<int32_t>(); auto pos = pos_.Get<int32_t>();
        Muls(skey, skey, -1.0f/static_cast<float>(stride_), n_);
        Cast(expert, skey, RoundMode::CAST_FLOOR, n_);
        Muls(expert, expert, int32_t(4), n_);
        Gather(gathered, starts, expert.ReinterpretCast<uint32_t>(), 0, count_);
        ArithProgression(pos, int32_t(0), int32_t(1), n_);
        Sub(gathered, pos, gathered, count_);
        Muls(sidx.ReinterpretCast<int32_t>(), sidx.ReinterpretCast<int32_t>(), int32_t(4), count_);
        Scatter(rank, gathered, sidx, 0, count_);
        Edge<HardEvent::V_MTE3>(pipe_);
        Edge<HardEvent::S_MTE3>(pipe_);
        DataCopyPad(rankGm_[begin_], rank, cp);
        DataCopy(countGm_[GetBlockIdx()*E], counts, E);
    }
};

class PrefixOffsets {
    TPipe pipe_;
    TBuf<TPosition::VECCALC> matrix_, total_, before_, offsets_, total64_;
    GlobalTensor<int32_t> countGm_, offsetGm_;
    GlobalTensor<int64_t> totalGm_;
public:
    __aicore__ inline void Init(GM_ADDR counts, GM_ADDR offsets, GM_ADDR totals) {
        countGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counts));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
        totalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(totals));
        pipe_.InitBuffer(matrix_, E*E*4); pipe_.InitBuffer(total_, E*4);
        pipe_.InitBuffer(before_, E*4); pipe_.InitBuffer(offsets_, E*4); pipe_.InitBuffer(total64_, E*8);
    }
    __aicore__ inline void Run() {
        auto matrix = matrix_.Get<int32_t>(); auto total = total_.Get<int32_t>();
        auto before = before_.Get<int32_t>(); auto offsets = offsets_.Get<int32_t>();
        auto total64 = total64_.Get<int64_t>();
        DataCopy(matrix, countGm_, E*E);
        Edge<HardEvent::MTE2_V>(pipe_);
        Duplicate(total, int32_t(0), E); Duplicate(before, int32_t(0), E);
        for (uint32_t c = 0; c < E; ++c) {
            Add(total, total, matrix[c*E], E);
            if (c < GetBlockIdx()) Add(before, before, matrix[c*E], E);
        }
        Edge<HardEvent::V_S>(pipe_);
        int32_t prefix = 0;
        for (uint32_t e = 0; e < E; ++e) {
            auto count = total.GetValue(e);
            offsets.SetValue(e, prefix + before.GetValue(e));
            if (GetBlockIdx() == 0) total64.SetValue(e, static_cast<int64_t>(count));
            prefix += count;
        }
        Edge<HardEvent::S_MTE3>(pipe_);
        DataCopy(offsetGm_[GetBlockIdx()*E], offsets, E);
        if (GetBlockIdx() == 0) DataCopy(totalGm_, total64, E);
    }
};

class ScatterPayload {
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> xQ_, scaleQ_;
    TBuf<TPosition::VECCALC> ids_, ranks_, offsets_, dst_;
    GlobalTensor<uint8_t> xGm_, scaleGm_, outGm_, outScaleGm_;
    GlobalTensor<int32_t> idsGm_, rankGm_, offsetGm_, mapGm_;
    uint32_t begin_, tokens_, tile_, slots_;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR ids, GM_ADDR ranks, GM_ADDR offsets,
                                GM_ADDR out, GM_ADDR outScale, GM_ADDR map,
                                uint32_t tokens, uint32_t perCore, uint32_t n, uint32_t tile) {
        begin_ = GetBlockIdx()*perCore; tokens_ = Min(perCore, tokens-begin_);
        slots_ = tokens_*K; tile_ = tile;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(scale));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(out));
        outScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outScale));
        idsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        rankGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranks));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
        mapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(map));
        pipe_.InitBuffer(xQ_, 2, tile*H); pipe_.InitBuffer(scaleQ_, 2, tile*S);
        pipe_.InitBuffer(ids_, n*4); pipe_.InitBuffer(ranks_, n*4);
        pipe_.InitBuffer(offsets_, E*4); pipe_.InitBuffer(dst_, n*4);
    }
    __aicore__ inline void CopyIn(uint32_t off, uint32_t n) {
        auto x = xQ_.AllocTensor<uint8_t>(); auto s = scaleQ_.AllocTensor<uint8_t>();
        DataCopy(x, xGm_[(begin_+off)*H], n*H);
        DataCopy(s, scaleGm_[(begin_+off)*S], n*S);
        xQ_.EnQue(x); scaleQ_.EnQue(s);
    }
    __aicore__ inline void CopyOut(uint32_t off, uint32_t n, const LocalTensor<int32_t>& dst) {
        auto x = xQ_.DeQue<uint8_t>(); auto s = scaleQ_.DeQue<uint8_t>();
        Edge<HardEvent::MTE2_MTE3>(pipe_);
        for (uint32_t t = 0; t < n; ++t) {
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t d = static_cast<uint32_t>(dst.GetValue((off+t)*K+k));
                DataCopy(outGm_[static_cast<uint64_t>(d)*H], x[t*H], H);
                DataCopy(outScaleGm_[static_cast<uint64_t>(d)*S], s[t*S], S);
            }
        }
        xQ_.FreeTensor(x); scaleQ_.FreeTensor(s);
    }
    __aicore__ inline void Run() {
        auto ids = ids_.Get<int32_t>(); auto rank = ranks_.Get<int32_t>();
        auto offsets = offsets_.Get<int32_t>(); auto dst = dst_.Get<int32_t>();
        DataCopyExtParams cp{1, slots_*4, 0, 0, 0}; DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(ids, idsGm_[begin_*K], cp, pad);
        DataCopyPad(rank, rankGm_[begin_*K], cp, pad);
        DataCopy(offsets, offsetGm_[GetBlockIdx()*E], E);
        Edge<HardEvent::MTE2_V>(pipe_);
        Muls(ids, ids, int32_t(4), slots_);
        Gather(dst, offsets, ids.ReinterpretCast<uint32_t>(), 0, slots_);
        Add(dst, dst, rank, slots_);
        Edge<HardEvent::V_MTE3>(pipe_);
        DataCopyPad(mapGm_[begin_*K], dst, cp);
        Edge<HardEvent::V_S>(pipe_);
        CopyIn(0, Min(tile_, tokens_));
        for (uint32_t off = 0; off < tokens_; off += tile_) {
            uint32_t next = off + tile_;
            if (next < tokens_) CopyIn(next, Min(tile_, tokens_-next));
            CopyOut(off, Min(tile_, tokens_-off), dst);
        }
    }
};
}

extern "C" __global__ __aicore__ void moe_local_ranks(
    GM_ADDR ids, GM_ADDR ranks, GM_ADDR counts, uint32_t tokens, uint32_t perCore,
    uint32_t sortSlots, uint32_t stride) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LocalRanks op; op.Init(ids, ranks, counts, tokens, perCore, sortSlots, stride); op.Run();
}
extern "C" __global__ __aicore__ void moe_prefix_offsets(
    GM_ADDR counts, GM_ADDR offsets, GM_ADDR totals) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    PrefixOffsets op; op.Init(counts, offsets, totals); op.Run();
}
extern "C" __global__ __aicore__ void moe_scatter_payload(
    GM_ADDR x, GM_ADDR scale, GM_ADDR ids, GM_ADDR ranks, GM_ADDR offsets,
    GM_ADDR out, GM_ADDR outScale, GM_ADDR map, uint32_t tokens, uint32_t perCore,
    uint32_t sortSlots, uint32_t tile) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ScatterPayload op; op.Init(x, scale, ids, ranks, offsets, out, outScale, map,
                              tokens, perCore, sortSlots, tile); op.Run();
}
