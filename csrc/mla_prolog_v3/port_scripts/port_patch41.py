#!/usr/bin/env python3
"""Register mla_prolog_v3_out (caller-provided output buffers)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P_H = os.path.join(ROOT, "include/sgl_kenel_npu_ops.h")
with io.open(P_H, encoding="utf-8") as f:
    s = f.read()
OLD = """std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache("""
NEW = """// Same operator, but writes into caller-provided buffers (stable addresses for the
// serving stack / ACL graph capture).
void mla_prolog_v3_out(const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                       const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
                       const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv,
                       const at::Tensor &rope_sin, const at::Tensor &rope_cos, at::Tensor &kv_cache,
                       at::Tensor &kr_cache, at::Tensor &query, at::Tensor &query_rope,
                       at::Tensor &dequant_q_nope, at::Tensor &query_norm, at::Tensor &dequant_q_norm,
                       const c10::optional<at::Tensor> &cache_index,
                       const c10::optional<at::Tensor> &dequant_scale_x,
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
                       int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size,
                       double qc_qr_scale, double kc_scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache("""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P_H, "w", encoding="utf-8") as f:
    f.write(s)

P_C = os.path.join(ROOT, "csrc/pytorch_extensions.cpp")
with io.open(P_C, encoding="utf-8") as f:
    s = f.read()

OLD_DEF = """        "float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
#endif"""
NEW_DEF = """        "float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def(
        "mla_prolog_v3_out(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, "
        "Tensor weight_dkv_kr, Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, "
        "Tensor(a!) kv_cache, Tensor(b!) kr_cache, Tensor(c!) query, Tensor(d!) query_rope, "
        "Tensor(e!) dequant_scale_q_nope, Tensor(f!) query_norm, Tensor(g!) dequant_scale_q_norm, "
        "*, Tensor? cache_index=None, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, "
        "Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, "
        "Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, "
        "Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, "
        "str? cache_mode=None, bool query_norm_flag=False, int weight_quant_mode=0, int kv_cache_quant_mode=0, "
        "int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, "
        "float qc_qr_scale=1.0, float kc_scale=1.0) -> ()");
#endif"""
assert s.count(OLD_DEF) == 1
s = s.replace(OLD_DEF, NEW_DEF)

OLD_IMPL = """    m.impl("mla_prolog_v3", TORCH_FN(sglang::npu_kernel::mla_prolog_v3));
#endif"""
NEW_IMPL = """    m.impl("mla_prolog_v3", TORCH_FN(sglang::npu_kernel::mla_prolog_v3));
    m.impl("mla_prolog_v3_out", TORCH_FN(sglang::npu_kernel::mla_prolog_v3_out));
#endif"""
assert s.count(OLD_IMPL) == 1
s = s.replace(OLD_IMPL, NEW_IMPL)
with io.open(P_C, "w", encoding="utf-8") as f:
    f.write(s)
print("mla_prolog_v3_out registered")
