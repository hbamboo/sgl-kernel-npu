#!/usr/bin/env python3
"""Port patch #6: wire MlaPrologV3 into the sgl_kernel_npu extension.

* declaration in include/sgl_kenel_npu_ops.h
* torch schema + PrivateUse1 impl registration (A5 only)
* CMake: kernel target, host sources, include dirs
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def patch(rel, pairs):
    path = os.path.join(ROOT, rel)
    with io.open(path, encoding="utf-8") as f:
        s = f.read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit("pattern count %d in %s for %r" % (n, rel, old[:70]))
        s = s.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(s)
    print("patched", rel)


# ---------------------------------------------------------------- declaration
patch("include/sgl_kenel_npu_ops.h", [
    ("""std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache(""",
     """// [sgl-kernel-npu port] MlaPrologV3 (GLM-4.7-Flash adapted, MXFP8 capable)
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
    int64_t ckvkr_repo_mode, int64_t quant_scale_repo_mode, int64_t tile_size, double qc_qr_scale, double kc_scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache("""),
])

# ---------------------------------------------------------------- registration
patch("csrc/pytorch_extensions.cpp", [
    ("""#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
    m.def(
        "kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
        "int quant_group_size, int quant_mode, bool round_scale_flag, int layout) -> ()");
#endif""",
     """#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
    m.def(
        "kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, "
        "int quant_group_size, int quant_mode, bool round_scale_flag, int layout) -> ()");

    m.def(
        "mla_prolog_v3(Tensor token_x, Tensor weight_dq, Tensor weight_uq_qr, Tensor weight_uk, Tensor weight_dkv_kr, "
        "Tensor rmsnorm_gamma_cq, Tensor rmsnorm_gamma_ckv, Tensor rope_sin, Tensor rope_cos, "
        "Tensor(a!) kv_cache, Tensor(b!) kr_cache, "
        "*, Tensor? cache_index=None, Tensor? dequant_scale_x=None, Tensor? dequant_scale_w_dq=None, "
        "Tensor? dequant_scale_w_uq_qr=None, Tensor? dequant_scale_w_dkv_kr=None, Tensor? quant_scale_ckv=None, "
        "Tensor? quant_scale_ckr=None, Tensor? smooth_scales_cq=None, Tensor? actual_seq_len=None, "
        "Tensor? k_nope_clip_alpha=None, float rmsnorm_epsilon_cq=1e-05, float rmsnorm_epsilon_ckv=1e-05, "
        "str? cache_mode=None, bool query_norm_flag=False, int weight_quant_mode=0, int kv_cache_quant_mode=0, "
        "int query_quant_mode=0, int ckvkr_repo_mode=0, int quant_scale_repo_mode=0, int tile_size=128, "
        "float qc_qr_scale=1.0, float kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
#endif"""),
    ("""#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
    m.impl("kv_compress_epilog", TORCH_FN(sglang::npu_kernel::kv_compress_epilog));
#endif""",
     """#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
    m.impl("kv_compress_epilog", TORCH_FN(sglang::npu_kernel::kv_compress_epilog));
    m.impl("mla_prolog_v3", TORCH_FN(sglang::npu_kernel::mla_prolog_v3));
#endif"""),
])

# ---------------------------------------------------------------- cmake
patch("csrc/CMakeLists.txt", [
    # host sources
    ("""# A5-only host sources
if(SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/kv_compress_epilog/op_host/kv_compress_epilog.cpp
    )
endif()""",
     """# A5-only host sources
if(SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    list(APPEND OP_SRCS
        ${PROJECT_OP_SRC_BASE}/kv_compress_epilog/op_host/kv_compress_epilog.cpp
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_host/mla_prolog_tiling.cpp
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_host/mla_prolog_tiling_check.cpp
    )
endif()"""),
    # kernel target
    ("""# Dedicated A3-only kernel targets. Keep these separate from workspace_kernel:""",
     """# A5-only MlaPrologV3 kernel: single named entry, runtime dispatch on tiling key.
set(A5_ONLY_KERNEL_TARGETS)
if(SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    ascendc_library(mla_prolog_v3_kernel STATIC
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp
    )
    ascendc_include_directories(mla_prolog_v3_kernel PRIVATE
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_kernel
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_kernel/arch35
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/op_kernel
        ${PROJECT_OP_SRC_BASE}/utils/kernel
    )
    ascendc_compile_definitions(mla_prolog_v3_kernel PRIVATE
        -DHAVE_WORKSPACE
        -DHAVE_TILING
    )
    list(APPEND A5_ONLY_KERNEL_TARGETS mla_prolog_v3_kernel)
endif()

# Dedicated A3-only kernel targets. Keep these separate from workspace_kernel:"""),
    # link
    ("""target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        ${A3_ONLY_KERNEL_TARGETS}""",
     """target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        ${A3_ONLY_KERNEL_TARGETS}
        ${A5_ONLY_KERNEL_TARGETS}"""),
    # include dirs
    ("""        ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host/tiling
        ${TORCH_DIR}/include""",
     """        ${PROJECT_OP_SRC_BASE}/sparse_attention_score/op_host/tiling
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/op_host
        ${PROJECT_OP_SRC_BASE}/mla_prolog_v3/compat
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_host
        ${PROJECT_OP_SRC_BASE}/mla_prolog/op_kernel
        ${ASCEND_INCLUDE_DIR}/op_common
        ${TORCH_DIR}/include"""),
])
print("done")
