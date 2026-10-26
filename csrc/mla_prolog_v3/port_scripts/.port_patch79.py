"""Split Launch into LaunchKernel + Launch, and add the fast path to mla_prolog_v3_out."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

# 1) split the EXEC part out of Launch
OLD1 = """void Launch(const MlaPrologV3Tensors &t, const at::Tensor &token_x, const at::Tensor &weight_dq,
            const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
            const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin,
            const at::Tensor &rope_cos, at::Tensor &kv_cache, at::Tensor &kr_cache, const OptionalInputs &opt,
            Outputs &out)
{
    optiling::MlaPrologTilingData tilingData{};
    MlaPrologV3TilingResult res;
    std::string err;
    uint64_t tilingKey = 0;
    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err, &tilingKey), "mla_prolog_v3 tiling failed: ",
                err);

    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    LaunchSlot &slot = SlotFor(tilingKey, device_index, tilingData, static_cast<int64_t>(res.workspaceSize), token_x);
    const at::Tensor &tiling_tensor = slot.tiling;
    at::Tensor &workspace = slot.workspace;

    const at::Tensor empty"""
NEW1 = """// Kernel launch only (no tiling): separated so the fast path can reuse a cached tiling.
void LaunchKernel(const at::Tensor &tiling_tensor, at::Tensor &workspace, uint32_t block_dim,
                  const at::Tensor &token_x, const at::Tensor &weight_dq, const at::Tensor &weight_uq_qr,
                  const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr, const at::Tensor &rmsnorm_gamma_cq,
                  const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                  at::Tensor &kv_cache, at::Tensor &kr_cache, const OptionalInputs &opt, Outputs &out)
{
    const at::Tensor empty"""
assert s.count(OLD1) == 1
s = s.replace(OLD1, NEW1)

OLD2 = """    // the launch macro captures its blockdim argument by name in a lambda, so it must
    // be a plain local variable
    const uint32_t block_dim = res.blockDim;
    EXEC_KERNEL_CMD(mla_prolog_v3, block_dim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_t,
                    scale_x_t, scale_wdq_t, scale_wuqqr_t, scale_wdkvkr_t, quant_ckv_t, quant_ckr_t, smooth_cq_t,
                    seq_len_t, clip_alpha_t, out.query, out.query_rope, kv_cache, kr_cache, out.dequant_q_nope,
                    out.query_norm, out.dequant_q_norm, workspace, tiling_tensor);
}"""
NEW2 = """    EXEC_KERNEL_CMD(mla_prolog_v3, block_dim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                    rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index_t,
                    scale_x_t, scale_wdq_t, scale_wuqqr_t, scale_wdkvkr_t, quant_ckv_t, quant_ckr_t, smooth_cq_t,
                    seq_len_t, clip_alpha_t, out.query, out.query_rope, kv_cache, kr_cache, out.dequant_q_nope,
                    out.query_norm, out.dequant_q_norm, workspace, tiling_tensor);
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
    uint64_t tilingKey = 0;
    TORCH_CHECK(MlaPrologV3RunTilingCached(t, &tilingData, &res, &err, &tilingKey), "mla_prolog_v3 tiling failed: ",
                err);

    int device_index = 0;
    c10_npu::GetDevice(&device_index);
    LaunchSlot &slot = SlotFor(tilingKey, device_index, tilingData, static_cast<int64_t>(res.workspaceSize), token_x);
    LaunchKernel(slot.tiling, slot.workspace, res.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                 rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
}"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("Launch split done")
