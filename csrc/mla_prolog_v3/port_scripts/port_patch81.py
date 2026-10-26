"""Fast path in mla_prolog_v3_out: reuse a validated entry, else fill one."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """    const Baseline b = DeriveBaseline(token_x, weight_dq, weight_uk, rope_sin, weight_quant_mode);
    TORCH_CHECK(query.numel() == b.tokens * b.n_heads * b.hckv, "query out buffer has the wrong size");
    TORCH_CHECK(query_rope.numel() == b.tokens * b.n_heads * b.dr, "query_rope out buffer has the wrong size");

    OptionalInputs opt{cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
                       dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
                       actual_seq_len, k_nope_clip_alpha};
    MlaPrologV3Tensors t = BuildTensors("""

NEW = """    const Baseline b = DeriveBaseline(token_x, weight_dq, weight_uk, rope_sin, weight_quant_mode);
    TORCH_CHECK(query.numel() == b.tokens * b.n_heads * b.hckv, "query out buffer has the wrong size");
    TORCH_CHECK(query_rope.numel() == b.tokens * b.n_heads * b.dr, "query_rope out buffer has the wrong size");

    OptionalInputs opt{cache_index, dequant_scale_x, dequant_scale_w_dq, dequant_scale_w_uq_qr,
                       dequant_scale_w_dkv_kr, quant_scale_ckv, quant_scale_ckr, smooth_scales_cq,
                       actual_seq_len, k_nope_clip_alpha};

    const std::string mode =
        cache_mode.has_value() ? std::string(cache_mode->data(), cache_mode->size()) : std::string("PA_BSND");
    int device_index = 0;
    c10_npu::GetDevice(&device_index);

    // Fast path: identical shapes/dtypes/attrs as one of the last launches -> reuse its
    // tiling blob and scratch buffer, skipping the description rebuild and the tiling.
    {
        static thread_local FastEntry entries[kFastEntries];
        static thread_local size_t nextSlot = 0;
        for (size_t i = 0; i < kFastEntries; ++i) {
            FastEntry &e = entries[i];
            if (!e.valid) {
                continue;
            }
            if (!Matches(e, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq,
                         rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, query, query_rope, query_norm,
                         dequant_q_nope, dequant_q_norm, opt, rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv, mode,
                         query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode,
                         quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, device_index)) {
                continue;
            }
            Outputs out{query, query_rope, dequant_q_nope, query_norm, dequant_q_norm};
            LaunchKernel(e.tiling, e.workspace, e.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                         rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
            return;
        }
        nextSlot = (nextSlot + 1) % kFastEntries;
    }

    MlaPrologV3Tensors t = BuildTensors("""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """    Outputs out{query, query_rope, dequant_q_nope, query_norm, dequant_q_norm};
    Launch(t, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
           rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
}"""
NEW2 = """    Outputs out{query, query_rope, dequant_q_nope, query_norm, dequant_q_norm};
    FastEntry *slotToFill = nullptr;
    {
        static thread_local FastEntry entries[kFastEntries];
        static thread_local size_t nextSlot = 0;
        slotToFill = &entries[nextSlot];
        nextSlot = (nextSlot + 1) % kFastEntries;
        slotToFill->valid = false;
    }
    Launch(t, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
           rope_sin, rope_cos, kv_cache, kr_cache, opt, out, slotToFill);
}"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("fast path wired into mla_prolog_v3_out")
