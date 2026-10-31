// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// PyBind11 entry points for the ported MlaPrologV3 kernels.
//
// The torch-library path (`torch.ops.npu.mla_prolog_v3`) spends ~16.7 us per call in
// the dispatcher alone - half of the decode latency.  These bindings let a serving
// stack call the same implementation directly and skip it:
//
//     import libsgl_kernel_npu as _C
//     _C.mla_prolog_v3_out(x, w_dq, w_uqqr, w_uk, w_dkv, g_cq, g_ckv, sin, cos,
//                          kv_cache, kr_cache, q, q_rope, dq_nope, q_norm, dq_norm, ...)
//
// The optional arguments mirror the torch schema; ``cache_mode`` is a plain string here
// (default "PA_BSND") to avoid string_view casting issues.

#include <string>

#include <torch/extension.h>

#include "sgl_kenel_npu_ops.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME libsgl_kernel_npu
#endif

namespace {

using OptTensor = c10::optional<at::Tensor>;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_prolog_v3_py(
    const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
    const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr, const at::Tensor &rmsnorm_gamma_cq,
    const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin, const at::Tensor &rope_cos,
    at::Tensor &kv_cache, at::Tensor &kr_cache, const OptTensor &cache_index, const OptTensor &dequant_scale_x,
    const OptTensor &dequant_scale_w_dq, const OptTensor &dequant_scale_w_uq_qr,
    const OptTensor &dequant_scale_w_dkv_kr, const OptTensor &quant_scale_ckv, const OptTensor &quant_scale_ckr,
    const OptTensor &smooth_scales_cq, const OptTensor &actual_seq_len, const OptTensor &k_nope_clip_alpha,
    double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv, const std::string &cache_mode, bool query_norm_flag,
    int64_t weight_quant_mode, int64_t kv_cache_quant_mode, int64_t query_quant_mode, int64_t ckvkr_repo_mode,
    int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    const c10::optional<c10::string_view> mode = c10::string_view(cache_mode);
    return sglang::npu_kernel::mla_prolog_v3(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin,
        rope_cos, kv_cache, kr_cache, cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
        dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, actual_seq_len,
        k_nope_clip_alpha, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, mode, query_norm_flag, weight_quant_mode,
        kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale,
        kc_scale);
}

void mla_prolog_v3_out_py(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                          const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
                          const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
                          const at::Tensor &rope_sin, const at::Tensor &rope_cos, at::Tensor &kv_cache,
                          at::Tensor &kr_cache, at::Tensor &query, at::Tensor &query_rope, at::Tensor &dequant_q_nope,
                          at::Tensor &query_norm, at::Tensor &dequant_q_norm, const OptTensor &cache_index,
                          const OptTensor &dequant_scale_x, const OptTensor &dequant_scale_w_dq,
                          const OptTensor &dequant_scale_w_uq_qr, const OptTensor &dequant_scale_w_dkv_kr,
                          const OptTensor &quant_scale_ckv, const OptTensor &quant_scale_ckr,
                          const OptTensor &smooth_scales_cq, const OptTensor &actual_seq_len,
                          const OptTensor &k_nope_clip_alpha, double rmsnorm_epsilon_cq, double rmsnorm_epsilon_ckv,
                          const std::string &cache_mode, bool query_norm_flag, int64_t weight_quant_mode,
                          int64_t kv_cache_quant_mode, int64_t query_quant_mode, int64_t ckvkr_repo_mode,
                          int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale)
{
    const c10::optional<c10::string_view> mode = c10::string_view(cache_mode);
    sglang::npu_kernel::mla_prolog_v3_out(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin,
        rope_cos, kv_cache, kr_cache, query, query_rope, dequant_q_nope, query_norm, dequant_q_norm, cache_index,
        dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr, quant_scale_ckv,
        quant_scale_ckr, smooth_scales_cq, actual_seq_len, k_nope_clip_alpha, rmsnorm_epsilon_cq,
        rmsnorm_epsilon_ckv, mode, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode,
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.attr("mla_prolog_v3_direct_cache_version") = 1;
    m.attr("mla_prolog_v3_packed_query_version") = 1;
    m.attr("mla_prolog_v3_small_decode_version") = 1;
    m.def("mla_prolog_v3", &mla_prolog_v3_py, "MlaPrologV3 (allocates output tensors)",
          py::arg("token_x"), py::arg("weight_dq"), py::arg("weight_uq_qr"), py::arg("weight_uk"),
          py::arg("weight_dkv_kr"), py::arg("rmsnorm_gamma_cq"), py::arg("rmsnorm_gamma_ckv"),
          py::arg("rope_sin"), py::arg("rope_cos"), py::arg("kv_cache"), py::arg("kr_cache"),
          py::arg("cache_index") = py::none(), py::arg("dequant_scale_x") = py::none(),
          py::arg("dequant_scale_w_dq") = py::none(), py::arg("dequant_scale_w_uq_qr") = py::none(),
          py::arg("dequant_scale_w_dkv_kr") = py::none(), py::arg("quant_scale_ckv") = py::none(),
          py::arg("quant_scale_ckr") = py::none(), py::arg("smooth_scales_cq") = py::none(),
          py::arg("actual_seq_len") = py::none(), py::arg("k_nope_clip_alpha") = py::none(),
          py::arg("rmsnorm_epsilon_cq") = 1e-05, py::arg("rmsnorm_epsilon_ckv") = 1e-05,
          py::arg("cache_mode") = "PA_BSND", py::arg("query_norm_flag") = false,
          py::arg("weight_quant_mode") = 0, py::arg("kv_cache_quant_mode") = 0,
          py::arg("query_quant_mode") = 0, py::arg("ckvkr_repo_mode") = 0,
          py::arg("quant_scale_repo_mode") = 0, py::arg("tile_size") = 128, py::arg("qc_qr_scale") = 1.0,
          py::arg("kc_scale") = 1.0);

    m.def("mla_prolog_v3_out", &mla_prolog_v3_out_py, "MlaPrologV3 into caller-provided buffers",
          py::arg("token_x"), py::arg("weight_dq"), py::arg("weight_uq_qr"), py::arg("weight_uk"),
          py::arg("weight_dkv_kr"), py::arg("rmsnorm_gamma_cq"), py::arg("rmsnorm_gamma_ckv"),
          py::arg("rope_sin"), py::arg("rope_cos"), py::arg("kv_cache"), py::arg("kr_cache"), py::arg("query"),
          py::arg("query_rope"), py::arg("dequant_q_nope"), py::arg("query_norm"), py::arg("dequant_q_norm"),
          py::arg("cache_index") = py::none(), py::arg("dequant_scale_x") = py::none(),
          py::arg("dequant_scale_w_dq") = py::none(), py::arg("dequant_scale_w_uq_qr") = py::none(),
          py::arg("dequant_scale_w_dkv_kr") = py::none(), py::arg("quant_scale_ckv") = py::none(),
          py::arg("quant_scale_ckr") = py::none(), py::arg("smooth_scales_cq") = py::none(),
          py::arg("actual_seq_len") = py::none(), py::arg("k_nope_clip_alpha") = py::none(),
          py::arg("rmsnorm_epsilon_cq") = 1e-05, py::arg("rmsnorm_epsilon_ckv") = 1e-05,
          py::arg("cache_mode") = "PA_BSND", py::arg("query_norm_flag") = false,
          py::arg("weight_quant_mode") = 0, py::arg("kv_cache_quant_mode") = 0,
          py::arg("query_quant_mode") = 0, py::arg("ckvkr_repo_mode") = 0,
          py::arg("quant_scale_repo_mode") = 0, py::arg("tile_size") = 128, py::arg("qc_qr_scale") = 1.0,
          py::arg("kc_scale") = 1.0);
}
