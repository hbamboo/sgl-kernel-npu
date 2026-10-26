#!/usr/bin/env python3
"""Replace the optional-input lambda with explicit tensors (the launch macro
expands into a std::apply that does not accept call expressions)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

OLD = """    // Optional inputs that were not provided are passed as empty tensors.
    const at::Tensor empty = EmptyLikeShape({0}, at::kFloat, token_x);
    auto pass = [&](const c10::optional<at::Tensor> &o) -> const at::Tensor & { return o.has_value() ? *o : empty; };

    EXEC_KERNEL_CMD(mla_prolog_v3, res.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache,
                    pass(cache_index), pass(dequant_scale_x), pass(dequant_scale_w_dq), pass(dequant_scale_w_uq_qr),
                    pass(dequant_scale_w_dkv_kr), pass(quant_scale_ckv), pass(quant_scale_ckr), pass(smooth_scales_cq),
                    pass(actual_seq_len), pass(k_nope_clip_alpha), query, query_rope, kv_cache, kr_cache,
                    dequant_q_nope, query_norm, dequant_q_norm, workspace, tiling_tensor);"""

NEW = """    // Optional inputs that were not provided are passed as empty tensors.
    const at::Tensor empty = EmptyLikeShape({0}, at::kFloat, token_x);
    at::Tensor cache_index_t = cache_index.has_value() ? cache_index.value() : empty;
    at::Tensor scale_x_t = dequant_scale_x.has_value() ? dequant_scale_x.value() : empty;
    at::Tensor scale_wdq_t = dequant_scale_w_dq.has_value() ? dequant_scale_w_dq.value() : empty;
    at::Tensor scale_wuqqr_t = dequant_scale_w_uq_qr.has_value() ? dequant_scale_w_uq_qr.value() : empty;
    at::Tensor scale_wdkvkr_t = dequant_scale_w_dkv_kr.has_value() ? dequant_scale_w_dkv_kr.value() : empty;
    at::Tensor quant_ckv_t = quant_scale_ckv.has_value() ? quant_scale_ckv.value() : empty;
    at::Tensor quant_ckr_t = quant_scale_ckr.has_value() ? quant_scale_ckr.value() : empty;
    at::Tensor smooth_cq_t = smooth_scales_cq.has_value() ? smooth_scales_cq.value() : empty;
    at::Tensor seq_len_t = actual_seq_len.has_value() ? actual_seq_len.value() : empty;
    at::Tensor clip_alpha_t = k_nope_clip_alpha.has_value() ? k_nope_clip_alpha.value() : empty;

    EXEC_KERNEL_CMD(mla_prolog_v3, res.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_t,
                    scale_x_t, scale_wdq_t, scale_wuqqr_t, scale_wdkvkr_t, quant_ckv_t, quant_ckr_t, smooth_cq_t,
                    seq_len_t, clip_alpha_t, query, query_rope, kv_cache, kr_cache, dequant_q_nope, query_norm,
                    dequant_q_norm, workspace, tiling_tensor);"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s.replace(OLD, NEW))
print("patched launch")
