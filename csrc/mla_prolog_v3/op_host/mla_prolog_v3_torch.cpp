// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Torch entry of the ported MlaPrologV3 op (direct kernel launch).
// Mirrors the aclnnMlaPrologV3WeightNz signature used by SGLang.
//
// Two entry points share the same launch path:
//   * mla_prolog_v3      - allocates the five outputs
//   * mla_prolog_v3_out  - writes into caller-provided buffers (stable addresses for
//                          the serving stack / ACL graph capture)

#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_mla_prolog_v3.h"

#include "mla_prolog_v3_host.h"

namespace sglang {
namespace npu_kernel {
namespace {

ge::DataType ToGeDtype(at::ScalarType t)
{
    switch (t) {
        case at::kFloat: return ge::DT_FLOAT;
        case at::kHalf: return ge::DT_FLOAT16;
        case at::kBFloat16: return ge::DT_BF16;
        case at::kChar: return ge::DT_INT8;
        case at::kInt: return ge::DT_INT32;
        case at::kLong: return ge::DT_INT64;
        case at::kByte: return ge::DT_UINT8;
        case at::kFloat8_e4m3fn: return ge::DT_FLOAT8_E4M3FN;
        case at::kFloat8_e5m2: return ge::DT_FLOAT8_E5M2;
        default: return ge::DT_BF16;
    }
}

std::vector<int64_t> DimsOf(const at::Tensor &t)
{
    std::vector<int64_t> d;
    d.reserve(t.dim());
    for (int64_t i = 0; i < t.dim(); ++i) {
        d.push_back(t.size(i));
    }
    return d;
}

std::vector<int64_t> DimsOf(const c10::optional<at::Tensor> &t)
{
    return t.has_value() ? DimsOf(*t) : std::vector<int64_t>{};
}

at::Tensor EmptyLikeShape(const std::vector<int64_t> &dims, at::ScalarType dtype, const at::Tensor &ref)
{
    return at::empty(dims, at::TensorOptions().dtype(dtype).device(ref.device()));
}

uint64_t HashBlob(const void *data, size_t bytes, uint64_t seed = 1469598103934665603ULL)
{
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        seed ^= static_cast<uint64_t>(p[i]);
        seed *= 1099511628211ULL;
    }
    return seed;
}

// Tiling data buffer, cached per (tiling content, device) so that repeated launches
// (including ACL graph capture) reuse a stable device address.
at::Tensor &CachedTiling(const optiling::MlaPrologTilingData &tiling)
{
    static std::mutex mtx;
    static std::unordered_map<uint64_t, at::Tensor> cache;

    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    uint64_t key = HashBlob(&tiling, sizeof(tiling));
    key = HashBlob(&device_index, sizeof(device_index), key);

    std::lock_guard<std::mutex> guard(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto host = at::empty({static_cast<int64_t>(sizeof(tiling))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(host.data_ptr(), &tiling, sizeof(tiling));
    at::Tensor pinned = host.pin_memory();
    at::Tensor device = pinned.to(c10::Device(c10::DeviceType::PrivateUse1, device_index), pinned.scalar_type(),
                                  /*non_blocking=*/true, /*copy=*/true);
    return cache.emplace(key, std::move(device)).first->second;
}

// Scratch buffer on the current stream: one per-device allocation, grown on demand.
at::Tensor WorkspaceFor(int64_t bytes, const at::Tensor &ref)
{
    static std::mutex mtx;
    static std::unordered_map<int64_t, at::Tensor> cache;
    const int64_t dev_index = ref.device().index();
    std::lock_guard<std::mutex> guard(mtx);
    auto &slot = cache[dev_index];
    if (!slot.defined() || slot.numel() < bytes) {
        slot = at::empty({bytes}, at::TensorOptions().dtype(at::kByte).device(ref.device()));
    }
    return slot;
}

struct OptionalInputs {
    const c10::optional<at::Tensor> &cache_index;
    const c10::optional<at::Tensor> &dequant_scale_x;
    const c10::optional<at::Tensor> &dequant_scale_w_dq;
    const c10::optional<at::Tensor> &dequant_scale_w_uq_qr;
    const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr;
    const c10::optional<at::Tensor> &quant_scale_ckv;
    const c10::optional<at::Tensor> &quant_scale_ckr;
    const c10::optional<at::Tensor> &smooth_scales_cq;
    const c10::optional<at::Tensor> &actual_seq_len;
    const c10::optional<at::Tensor> &k_nope_clip_alpha;
};

struct Baseline {
    int64_t tokens;
    int64_t n_heads;
    int64_t hckv;
    int64_t dr;
    int64_t hcq;
    at::ScalarType qn_dtype;
};

Baseline DeriveBaseline(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uk,
                        const at::Tensor &rope_sin, int64_t weight_quant_mode)
{
    Baseline b{};
    b.tokens = token_x.dim() == 2 ? token_x.size(0) : token_x.size(0) * token_x.size(1);
    b.n_heads = weight_uk.size(0);
    b.hckv = weight_uk.size(2);
    b.dr = rope_sin.size(rope_sin.dim() - 1);
    b.hcq = weight_dq.size(weight_dq.dim() - 1);
    b.qn_dtype = at::kBFloat16;
    if (weight_quant_mode == 3 || weight_quant_mode == 4) {
        b.qn_dtype = at::kFloat8_e4m3fn;
    } else if (weight_quant_mode == 1 || weight_quant_mode == 2) {
        b.qn_dtype = at::kChar;
    }
    return b;
}

MlaPrologV3Tensors BuildTensors(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                                const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
                                const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
                                const at::Tensor &rope_sin, const at::Tensor &rope_cos, const at::Tensor &kv_cache,
                                const at::Tensor &kr_cache, const OptionalInputs &opt, const Baseline &b,
                                int64_t weight_quant_mode, int64_t kv_cache_quant_mode, bool query_norm_flag)
{
    MlaPrologV3Tensors t;
    t.tokenX = DimsOf(token_x);
    t.weightDq = DimsOf(weight_dq);
    t.weightUqQr = DimsOf(weight_uq_qr);
    t.weightUk = DimsOf(weight_uk);
    t.weightDkvKr = DimsOf(weight_dkv_kr);
    t.gammaCq = DimsOf(rmsnorm_gamma_cq);
    t.gammaCkv = DimsOf(rmsnorm_gamma_ckv);
    t.ropeSin = DimsOf(rope_sin);
    t.ropeCos = DimsOf(rope_cos);
    t.kvCache = DimsOf(kv_cache);
    t.krCache = DimsOf(kr_cache);
    t.cacheIndex = DimsOf(opt.cache_index);
    t.dequantScaleX = DimsOf(opt.dequant_scale_x);
    t.dequantScaleWDq = DimsOf(opt.dequant_scale_w_dq);
    t.dequantScaleWUqQr = DimsOf(opt.dequant_scale_w_uq_qr);
    t.dequantScaleWDkvKr = DimsOf(opt.dequant_scale_w_dkv_kr);
    t.quantScaleCkv = DimsOf(opt.quant_scale_ckv);
    t.quantScaleCkr = DimsOf(opt.quant_scale_ckr);
    t.smoothScalesCq = DimsOf(opt.smooth_scales_cq);
    t.actualSeqLen = DimsOf(opt.actual_seq_len);
    t.kNopeClipAlpha = DimsOf(opt.k_nope_clip_alpha);

    t.tokenXType = ToGeDtype(token_x.scalar_type());
    t.weightDqType = ToGeDtype(weight_dq.scalar_type());
    t.weightUqQrType = ToGeDtype(weight_uq_qr.scalar_type());
    t.weightUkType = ToGeDtype(weight_uk.scalar_type());
    t.weightDkvKrType = ToGeDtype(weight_dkv_kr.scalar_type());
    t.gammaCqType = ToGeDtype(rmsnorm_gamma_cq.scalar_type());
    t.gammaCkvType = ToGeDtype(rmsnorm_gamma_ckv.scalar_type());
    t.ropeSinType = ToGeDtype(rope_sin.scalar_type());
    t.ropeCosType = ToGeDtype(rope_cos.scalar_type());
    t.kvCacheType = ToGeDtype(kv_cache.scalar_type());
    t.krCacheType = ToGeDtype(kr_cache.scalar_type());

    auto optDtype = [&](const c10::optional<at::Tensor> &o) {
        return o.has_value() ? ToGeDtype(o->scalar_type()) : ge::DT_FLOAT;
    };
    t.cacheIndexType = opt.cache_index.has_value() ? ToGeDtype(opt.cache_index->scalar_type()) : ge::DT_INT64;
    t.scaleXType = optDtype(opt.dequant_scale_x);
    t.scaleWDqType = optDtype(opt.dequant_scale_w_dq);
    t.scaleWUqQrType = optDtype(opt.dequant_scale_w_uq_qr);
    t.scaleWDkvKrType = optDtype(opt.dequant_scale_w_dkv_kr);
    if (weight_quant_mode == 3) {
        // MXFP8: scale tensors are FLOAT8_E8M0, carried as uint8 on the torch side.
        t.scaleXType = ge::DT_FLOAT8_E8M0;
        t.scaleWDqType = ge::DT_FLOAT8_E8M0;
        t.scaleWUqQrType = ge::DT_FLOAT8_E8M0;
        t.scaleWDkvKrType = ge::DT_FLOAT8_E8M0;
        t.scaleQNormType = ge::DT_FLOAT8_E8M0;
    }
    t.actualSeqLenType =
        opt.actual_seq_len.has_value() ? ToGeDtype(opt.actual_seq_len->scalar_type()) : ge::DT_INT32;
    t.clipAlphaType =
        opt.k_nope_clip_alpha.has_value() ? ToGeDtype(opt.k_nope_clip_alpha->scalar_type()) : ge::DT_FLOAT;

    // ---- output shapes (the tiling validates them by scenario)
    t.query = {b.tokens, b.n_heads, b.hckv};
    t.queryRope = {b.tokens, b.n_heads, b.dr};
    t.kvCacheOut = t.kvCache;
    t.krCacheOut = t.krCache;
    t.dequantScaleQNope = {0};
    t.queryNormType = ToGeDtype(b.qn_dtype);
    if (query_norm_flag) {
        t.queryNorm = {b.tokens, b.hcq};
        if (weight_quant_mode == 0) {
            t.dequantScaleQNorm = {0};
            t.scaleQNormType = ge::DT_FLOAT;
        } else if (weight_quant_mode == 3) {
            t.dequantScaleQNorm = {b.tokens, b.hcq / 32};
            t.scaleQNormType = ge::DT_FLOAT8_E8M0;
        } else {
            t.dequantScaleQNorm = {b.tokens, 1};
            t.scaleQNormType = ge::DT_FLOAT;
        }
    } else {
        t.queryNorm = {0};
        t.dequantScaleQNorm = {0};
    }
    if (kv_cache_quant_mode == 1) {
        t.dequantScaleQNope = {b.tokens, b.n_heads, 1};
    }
    t.scaleQNopeType = ge::DT_FLOAT;
    return t;
}

struct Outputs {
    at::Tensor query;
    at::Tensor query_rope;
    at::Tensor dequant_q_nope;
    at::Tensor query_norm;
    at::Tensor dequant_q_norm;
};

Outputs AllocateOutputs(const at::Tensor &ref, const Baseline &b, int64_t weight_quant_mode,
                        int64_t kv_cache_quant_mode, bool query_norm_flag)
{
    Outputs o{};
    o.query = EmptyLikeShape({b.tokens, b.n_heads, b.hckv}, at::kBFloat16, ref);
    o.query_rope = EmptyLikeShape({b.tokens, b.n_heads, b.dr}, at::kBFloat16, ref);
    o.dequant_q_nope =
        EmptyLikeShape({kv_cache_quant_mode == 1 ? b.tokens : 0, b.n_heads, 1}, at::kFloat, ref);
    o.query_norm = query_norm_flag ? EmptyLikeShape({b.tokens, b.hcq}, b.qn_dtype, ref)
                                   : EmptyLikeShape({0}, b.qn_dtype, ref);
    if (!query_norm_flag || weight_quant_mode == 0) {
        o.dequant_q_norm = EmptyLikeShape({0}, at::kFloat, ref);
    } else if (weight_quant_mode == 3) {
        o.dequant_q_norm = EmptyLikeShape({b.tokens, b.hcq / 32}, at::kByte, ref);
    } else {
        o.dequant_q_norm = EmptyLikeShape({b.tokens, 1}, at::kFloat, ref);
    }
    return o;
}

void Launch(const MlaPrologV3Tensors &t, const at::Tensor &token_x, const at::Tensor &weight_dq,
            const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
            const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin,
            const at::Tensor &rope_cos, at::Tensor &kv_cache, at::Tensor &kr_cache, const OptionalInputs &opt,
            Outputs &out)
{
    optiling::MlaPrologTilingData tilingData{};
    MlaPrologV3TilingResult res;
    std::string err;
    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err), "mla_prolog_v3 tiling failed: ", err);

    const at::Tensor &tiling_tensor = CachedTiling(tilingData);
    at::Tensor workspace = WorkspaceFor(static_cast<int64_t>(res.workspaceSize), token_x);

    const at::Tensor empty = EmptyLikeShape({0}, at::kFloat, token_x);
    at::Tensor cache_index_t = opt.cache_index.has_value() ? opt.cache_index.value() : empty;
    at::Tensor scale_x_t = opt.dequant_scale_x.has_value() ? opt.dequant_scale_x.value() : empty;
    at::Tensor scale_wdq_t = opt.dequant_scale_w_dq.has_value() ? opt.dequant_scale_w_dq.value() : empty;
    at::Tensor scale_wuqqr_t = opt.dequant_scale_w_uq_qr.has_value() ? opt.dequant_scale_w_uq_qr.value() : empty;
    at::Tensor scale_wdkvkr_t = opt.dequant_scale_w_dkv_kr.has_value() ? opt.dequant_scale_w_dkv_kr.value() : empty;
    at::Tensor quant_ckv_t = opt.quant_scale_ckv.has_value() ? opt.quant_scale_ckv.value() : empty;
    at::Tensor quant_ckr_t = opt.quant_scale_ckr.has_value() ? opt.quant_scale_ckr.value() : empty;
    at::Tensor smooth_cq_t = opt.smooth_scales_cq.has_value() ? opt.smooth_scales_cq.value() : empty;
    at::Tensor seq_len_t = opt.actual_seq_len.has_value() ? opt.actual_seq_len.value() : empty;
    at::Tensor clip_alpha_t = opt.k_nope_clip_alpha.has_value() ? opt.k_nope_clip_alpha.value() : empty;

    // the launch macro captures its blockdim argument by name in a lambda, so it must
    // be a plain local variable
    const uint32_t block_dim = res.blockDim;
    EXEC_KERNEL_CMD(mla_prolog_v3, block_dim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_t,
                    scale_x_t, scale_wdq_t, scale_wuqqr_t, scale_wdkvkr_t, quant_ckv_t, quant_ckr_t, smooth_cq_t,
                    seq_len_t, clip_alpha_t, out.query, out.query_rope, kv_cache, kr_cache, out.dequant_q_nope,
                    out.query_norm, out.dequant_q_norm, workspace, tiling_tensor);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_prolog_v3(
    const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk,
    const at::Tensor &weight_dkv_kr, const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
    const at::Tensor &rope_sin, const at::Tensor &rope_cos, at::Tensor &kv_cache, at::Tensor &kr_cache,
    const c10::optional<at::Tensor> &cache_index, const c10::optional<at::Tensor> &dequant_scale_x,
    const c10::optional<at::Tensor> &dequant_scale_w_dq, const c10::optional<at::Tensor> &dequant_scale_w_uq_qr,
    const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr, const c10::optional<at::Tensor> &quant_scale_ckv,
    const c10::optional<at::Tensor> &quant_scale_ckr, const c10::optional<at::Tensor> &smooth_scales_cq,
    const c10::optional<at::Tensor> &actual_seq_len, const c10::optional<at::Tensor> &k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, c10::optional<c10::string_view> cache_mode,
    bool query_norm_flag, int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode,
    int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    TORCH_CHECK(token_x.dim() == 2 || token_x.dim() == 3, "token_x must be [T, He] or [B, S, He]");
    TORCH_CHECK(weight_uk.dim() == 3, "weight_uk must be [N, D, Hckv]");
    TORCH_CHECK(kv_cache.dim() == 4 && kr_cache.dim() == 4, "kv_cache/kr_cache must be 4D");

    const Baseline b = DeriveBaseline(token_x, weight_dq, weight_uk, rope_sin, weight_quant_mode);
    OptionalInputs opt{cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
                       dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
                       actual_seq_len, k_nope_clip_alpha};
    MlaPrologV3Tensors t = BuildTensors(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
                                        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, b,
                                        weight_quant_mode, kv_cache_quant_mode, query_norm_flag);
    t.rmsEpsCq = static_cast<float>(rmsnorm_epsilon_cq);
    t.rmsEpsCkv = static_cast<float>(rmsnorm_epsilon_ckv);
    t.cacheMode = cache_mode.has_value() ? std::string(cache_mode->data(), cache_mode->size()) : std::string("PA_BSND");
    t.queryNormFlag = query_norm_flag;
    t.weightQuantMode = weight_quant_mode;
    t.kvQuantMode = kv_cache_quant_mode;
    t.queryQuantMode = query_quant_mode;
    t.ckvkrRepoMode = ckvkr_repo_mode;
    t.quantScaleRepoMode = quant_scale_repo_mode;
    t.tileSize = tile_size;
    t.qcQrScale = static_cast<float>(qc_qr_scale);
    t.kcScale = static_cast<float>(kc_scale);

    Outputs out = AllocateOutputs(token_x, b, weight_quant_mode, kv_cache_quant_mode, query_norm_flag);
    Launch(t, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
           rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
    return std::make_tuple(out.query, out.query_rope, out.dequant_q_nope, out.query_norm, out.dequant_q_norm);
}

void mla_prolog_v3_out(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                       const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
                       const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
                       const at::Tensor &rope_sin, const at::Tensor &rope_cos, at::Tensor &kv_cache,
                       at::Tensor &kr_cache, at::Tensor &query, at::Tensor &query_rope,
                       at::Tensor &dequant_q_nope, at::Tensor &query_norm, at::Tensor &dequant_q_norm,
                       const c10::optional<at::Tensor> &cache_index, const c10::optional<at::Tensor> &dequant_scale_x,
                       const c10::optional<at::Tensor> &dequant_scale_w_dq,
                       const c10::optional<at::Tensor> &dequant_scale_w_uq_qr,
                       const c10::optional<at::Tensor> &dequant_scale_w_dkv_kr,
                       const c10::optional<at::Tensor> &quant_scale_ckv,
                       const c10::optional<at::Tensor> &quant_scale_ckr,
                       const c10::optional<at::Tensor> &smooth_scales_cq,
                       const c10::optional<at::Tensor> &actual_seq_len,
                       const c10::optional<at::Tensor> &k_nope_clip_alpha, double rmsnorm_epsilon_cq,
                       double rmsnorm_epsilon_ckv, c10::optional<c10::string_view> cache_mode, bool query_norm_flag,
                       int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode,
                       int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale,
                       double kc_scale)
{
    const Baseline b = DeriveBaseline(token_x, weight_dq, weight_uk, rope_sin, weight_quant_mode);
    TORCH_CHECK(query.numel() == b.tokens * b.n_heads * b.hckv, "query out buffer has the wrong size");
    TORCH_CHECK(query_rope.numel() == b.tokens * b.n_heads * b.dr, "query_rope out buffer has the wrong size");

    OptionalInputs opt{cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
                       dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
                       actual_seq_len, k_nope_clip_alpha};
    MlaPrologV3Tensors t = BuildTensors(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
                                        rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, b,
                                        weight_quant_mode, kv_cache_quant_mode, query_norm_flag);
    t.rmsEpsCq = static_cast<float>(rmsnorm_epsilon_cq);
    t.rmsEpsCkv = static_cast<float>(rmsnorm_epsilon_ckv);
    t.cacheMode = cache_mode.has_value() ? std::string(cache_mode->data(), cache_mode->size()) : std::string("PA_BSND");
    t.queryNormFlag = query_norm_flag;
    t.weightQuantMode = weight_quant_mode;
    t.kvQuantMode = kv_cache_quant_mode;
    t.queryQuantMode = query_quant_mode;
    t.ckvkrRepoMode = ckvkr_repo_mode;
    t.quantScaleRepoMode = quant_scale_repo_mode;
    t.tileSize = tile_size;
    t.qcQrScale = static_cast<float>(qc_qr_scale);
    t.kcScale = static_cast<float>(kc_scale);
    // use the caller's buffers as the expected output shapes
    t.query = DimsOf(query);
    t.queryRope = DimsOf(query_rope);
    t.queryNorm = DimsOf(query_norm);
    t.dequantScaleQNorm = DimsOf(dequant_q_norm);
    t.dequantScaleQNope = DimsOf(dequant_q_nope);

    Outputs out{query, query_rope, dequant_q_nope, query_norm, dequant_q_norm};
    Launch(t, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
           rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
}

}  // namespace npu_kernel
}  // namespace sglang
