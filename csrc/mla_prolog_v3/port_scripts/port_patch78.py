"""Torch layer: allocation-free fast path for repeated launches.

Measured: the pybind call spends 8.3 us in our host code before the kernel launch
(11.8 us total minus ~3.4 us launch), and the dominant part is rebuilding the
`MlaPrologV3Tensors` description (20 `std::vector` allocations per call) plus the tiling
lookup.  The description depends only on shapes/dtypes/attrs, so keep the last few
launches in a thread-local ring and revalidate them against the tensors without
allocating anything.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

ANCHOR = """void Launch(const MlaPrologV3Tensors &t, const at::Tensor &token_x, const at::Tensor &weight_dq,"""

FAST = '''// ---------------------------------------------------------------------------
// Allocation-free fast path: the launch description depends only on shapes, dtypes and
// attributes, so a repeated launch (same layer, same step) can reuse the tiling blob and
// the scratch buffer after revalidating the shapes - without building the 20-odd
// std::vector fields of MlaPrologV3Tensors again.
bool SameShape(const std::vector<int64_t> &dims, const at::Tensor &t)
{
    if (static_cast<int64_t>(dims.size()) != t.dim()) {
        return false;
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != t.size(static_cast<int64_t>(i))) {
            return false;
        }
    }
    return true;
}

bool SameShapeOpt(const std::vector<int64_t> &dims, const c10::optional<at::Tensor> &t)
{
    if (!t.has_value()) {
        return dims.empty();
    }
    return SameShape(dims, *t);
}

struct FastEntry {
    MlaPrologV3Tensors tensors;
    optiling::MlaPrologTilingData tilingData;
    at::Tensor tiling;
    at::Tensor workspace;
    int64_t workspaceBytes;
    int device;
    bool valid;
};

constexpr size_t kFastEntries = 16;

bool Matches(const FastEntry &e, const at::Tensor &token_x, const at::Tensor &weight_dq,
             const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
             const at::Tensor &gamma_cq, const at::Tensor &gamma_ckv, const at::Tensor &rope_sin,
             const at::Tensor &rope_cos, const at::Tensor &kv_cache, const at::Tensor &kr_cache,
             const at::Tensor &query, const at::Tensor &query_rope, const at::Tensor &query_norm,
             const at::Tensor &dequant_q_nope, const at::Tensor &dequant_q_norm, const OptionalInputs &opt,
             double eps_cq, double eps_ckv, const std::string &cache_mode, bool query_norm_flag,
             int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode,
             int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale,
             double kc_scale, int device)
{
    const MlaPrologV3Tensors &t = e.tensors;
    return e.device == device && SameShape(t.tokenX, token_x) && SameShape(t.weightDq, weight_dq) &&
           SameShape(t.weightUqQr, weight_uq_qr) && SameShape(t.weightUk, weight_uk) &&
           SameShape(t.weightDkvKr, weight_dkv_kr) && SameShape(t.gammaCq, gamma_cq) &&
           SameShape(t.gammaCkv, gamma_ckv) && SameShape(t.ropeSin, rope_sin) && SameShape(t.ropeCos, rope_cos) &&
           SameShape(t.kvCache, kv_cache) && SameShape(t.krCache, kr_cache) && SameShape(t.query, query) &&
           SameShape(t.queryRope, query_rope) && SameShapeOpt(t.queryNorm, query_norm_flag ? c10::optional<at::Tensor>(query_norm) : c10::nullopt) &&
           SameShapeOpt(t.dequantScaleQNope, dequant_q_nope.numel() ? c10::optional<at::Tensor>(dequant_q_nope) : c10::nullopt) &&
           SameShapeOpt(t.dequantScaleQNorm, dequant_q_norm.numel() ? c10::optional<at::Tensor>(dequant_q_norm) : c10::nullopt) &&
           SameShapeOpt(t.cacheIndex, opt.cache_index) && SameShapeOpt(t.dequantScaleX, opt.dequant_scale_x) &&
           SameShapeOpt(t.dequantScaleWDq, opt.dequant_scale_w_dq) &&
           SameShapeOpt(t.dequantScaleWUqQr, opt.dequant_scale_w_uq_qr) &&
           SameShapeOpt(t.dequantScaleWDkvKr, opt.dequant_scale_w_dkv_kr) &&
           SameShapeOpt(t.quantScaleCkv, opt.quant_scale_ckv) && SameShapeOpt(t.quantScaleCkr, opt.quant_scale_ckr) &&
           SameShapeOpt(t.smoothScalesCq, opt.smooth_scales_cq) &&
           SameShapeOpt(t.actualSeqLen, opt.actual_seq_len) &&
           SameShapeOpt(t.kNopeClipAlpha, opt.k_nope_clip_alpha) &&
           t.tokenXType == ToGeDtype(token_x.scalar_type()) &&
           t.weightDqType == ToGeDtype(weight_dq.scalar_type()) &&
           t.weightUqQrType == ToGeDtype(weight_uq_qr.scalar_type()) &&
           t.weightUkType == ToGeDtype(weight_uk.scalar_type()) &&
           t.weightDkvKrType == ToGeDtype(weight_dkv_kr.scalar_type()) &&
           t.rmsEpsCq == static_cast<float>(eps_cq) && t.rmsEpsCkv == static_cast<float>(eps_ckv) &&
           t.cacheMode == cache_mode && t.queryNormFlag == query_norm_flag &&
           t.weightQuantMode == weight_quant_mode && t.kvQuantMode == kv_cache_quant_mode &&
           t.queryQuantMode == query_quant_mode && t.ckvkrRepoMode == ckvkr_repo_mode &&
           t.quantScaleRepoMode == quant_scale_repo_mode && t.tileSize == tile_size &&
           t.qcQrScale == static_cast<float>(qc_qr_scale) && t.kcScale == static_cast<float>(kc_scale);
}

'''

assert s.count(ANCHOR) == 1
s = s.replace(ANCHOR, FAST + ANCHOR)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("fast-path helpers added")
