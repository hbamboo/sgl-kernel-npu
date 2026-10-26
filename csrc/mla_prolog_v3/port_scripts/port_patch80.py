"""Wire the fast path into mla_prolog_v3_out."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_torch.cpp")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD1 = """struct FastEntry {
    MlaPrologV3Tensors tensors;
    optiling::MlaPrologTilingData tilingData;
    at::Tensor tiling;
    at::Tensor workspace;
    int64_t workspaceBytes;
    int device;
    bool valid;
};"""
NEW1 = """struct FastEntry {
    MlaPrologV3Tensors tensors;
    at::Tensor tiling;
    at::Tensor workspace;
    uint32_t blockDim{0};
    int device{-1};
    bool valid{false};
};"""
assert s.count(OLD1) == 1
s = s.replace(OLD1, NEW1)

# Launch gains an optional cache-out
OLD2 = """void Launch(const MlaPrologV3Tensors &t, const at::Tensor &token_x, const at::Tensor &weight_dq,
            const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
            const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin,
            const at::Tensor &rope_cos, at::Tensor &kv_cache, at::Tensor &kr_cache, const OptionalInputs &opt,
            Outputs &out)
{"""
NEW2 = """void Launch(const MlaPrologV3Tensors &t, const at::Tensor &token_x, const at::Tensor &weight_dq,
            const at::Tensor &weight_uq_qr, const at::Tensor &weight_uk, const at::Tensor &weight_dkv_kr,
            const at::Tensor &rmsnorm_gamma_cq, const at::Tensor &rmsnorm_gamma_ckv, const at::Tensor &rope_sin,
            const at::Tensor &rope_cos, at::Tensor &kv_cache, at::Tensor &kr_cache, const OptionalInputs &opt,
            Outputs &out, FastEntry *cacheOut = nullptr)
{"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = """    LaunchKernel(slot.tiling, slot.workspace, res.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                 rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
}"""
NEW3 = """    if (cacheOut != nullptr) {
        cacheOut->tensors = t;
        cacheOut->tiling = slot.tiling;
        cacheOut->workspace = slot.workspace;
        cacheOut->blockDim = res.blockDim;
        cacheOut->device = device_index;
        cacheOut->valid = true;
    }
    LaunchKernel(slot.tiling, slot.workspace, res.blockDim, token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
                 rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, opt, out);
}"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("Launch cache-out wired")
