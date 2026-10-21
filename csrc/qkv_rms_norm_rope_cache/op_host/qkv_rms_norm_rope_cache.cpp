// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Host launcher for the AscendC qkv_rms_norm_rope_cache kernel.
//
// Tiling: three core groups (Q / K / V).  Q and K run the full RMSNorm+RoPE pipeline, V is
// a pure PA_NZ scatter, so the core budget is split by measured per-head cost
// (Q : K : V = 1.0 : 1.7 : 0.9).  See OPTIMIZATION.md for the measurements behind it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>

#include "defines.h"
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "sgl_kenel_npu_ops.h"
#include "qkv_rms_norm_rope_cache_tiling.h"
#include "aclrtlaunch_qkv_rms_norm_rope_cache.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr int64_t kBlockAlignNumFp16 = 16;
// Heads staged per UB tile by the RegBase kernel (it keeps only one input and one output
// staging buffer, so this can be much larger than the old UB-tile kernel allowed).
constexpr int64_t kUbRows = 192;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

QkvRmsNormRopeCacheTilingData BuildTiling(int64_t tokens, int64_t num_head_q, int64_t num_head_k,
                                          int64_t num_head_v, int64_t dim, int64_t block_size, float epsilon)
{
    QkvRmsNormRopeCacheTilingData tiling{};
    tiling.batchSize = tokens;
    tiling.seqLength = 1;
    tiling.numHead = num_head_q + num_head_k + num_head_v;
    tiling.qkvDim = dim;
    tiling.numHeadQ = num_head_q;
    tiling.numHeadK = num_head_k;
    tiling.numHeadV = num_head_v;
    tiling.blockSize = block_size;
    tiling.epsilon = epsilon;
    tiling.ubFactor = kUbRows;

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t core_num = static_cast<int64_t>(platform->GetCoreNumAiv());

    const int64_t token_sum_q = tokens * num_head_q;
    const int64_t token_sum_k = tokens * num_head_k;
    const int64_t token_sum_v = tokens * num_head_v;

    // Split by measured work, not by head count: an upstream-style head-count split
    // (K:V = 0.6:0.4 of a KV budget sized by heads) starves K, whose per-head cost is ~1.7x
    // a Q head because it also pays the PA_NZ scatter.  Reproduces the measured optimum
    // Q:K:V = 39:16:9 on a 64-core A950 for 32k+ tokens.
    constexpr double kWorkQ = 1.0;
    constexpr double kWorkK = 1.7;
    constexpr double kWorkV = 0.9;
    const double work_q = static_cast<double>(token_sum_q) * kWorkQ;
    const double work_kv = static_cast<double>(token_sum_k) * kWorkK +
                           static_cast<double>(token_sum_v) * kWorkV;
    int64_t cores_kv = static_cast<int64_t>(
        std::llround(static_cast<double>(core_num) * work_kv / (work_q + work_kv)));
    cores_kv = std::min<int64_t>(core_num - 1, std::max<int64_t>(2, cores_kv));
    int64_t cores_k = static_cast<int64_t>(
        std::llround(static_cast<double>(cores_kv) * static_cast<double>(token_sum_k) * kWorkK / work_kv));
    cores_k = std::min<int64_t>(cores_kv - 1, std::max<int64_t>(1, cores_k));
    const int64_t cores_v = cores_kv - cores_k;
    const int64_t cores_q = std::max<int64_t>(1, core_num - cores_k - cores_v);

    tiling.blockFactorQ = CeilDiv(token_sum_q, cores_q);
    tiling.blockDimQ = CeilDiv(token_sum_q, tiling.blockFactorQ);
    tiling.blockFactorK = CeilDiv(token_sum_k, cores_k);
    tiling.blockDimK = CeilDiv(token_sum_k, tiling.blockFactorK);
    tiling.blockFactorV = CeilDiv(token_sum_v, cores_v);
    tiling.blockDimV = CeilDiv(token_sum_v, tiling.blockFactorV);
    tiling.blockDim = tiling.blockDimQ + tiling.blockDimK + tiling.blockDimV;
    return tiling;
}

at::Tensor CopyTilingToDevice(const QkvRmsNormRopeCacheTilingData &tiling)
{
    constexpr int64_t kPadding = 32;
    const int64_t bytes =
        (static_cast<int64_t>(sizeof(QkvRmsNormRopeCacheTilingData)) + kPadding - 1) / kPadding * kPadding;
    auto buffer = at::empty({bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(buffer.data_ptr(), &tiling, sizeof(QkvRmsNormRopeCacheTilingData));
    return TorchNpuHelper::CopyTensorHostToDevice(buffer);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_rms_norm_rope_cache(
    const at::Tensor &qkv, const at::Tensor &q_gamma, const at::Tensor &k_gamma, const at::Tensor &cos,
    const at::Tensor &sin, const at::Tensor &index, at::Tensor q_out, at::Tensor k_cache, at::Tensor v_cache,
    int64_t num_head_q, int64_t num_head_k, int64_t num_head_v, double epsilon)
{
    TORCH_CHECK(qkv.dim() == 2, "qkv must be a 2D tensor [tokens, N*D]");
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "qkv must be bfloat16");
    TORCH_CHECK(q_gamma.dim() == 1 && k_gamma.dim() == 1, "q_gamma/k_gamma must be 1D");
    TORCH_CHECK(q_gamma.numel() == k_gamma.numel(), "q_gamma and k_gamma must have the same size");
    TORCH_CHECK(q_gamma.scalar_type() == at::kBFloat16 && k_gamma.scalar_type() == at::kBFloat16,
                "q_gamma/k_gamma must be bfloat16");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos/sin must be 2D [tokens, D]");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16,
                "cos/sin must be bfloat16");
    TORCH_CHECK(index.dim() == 1 && index.scalar_type() == at::kLong, "index must be a 1D int64 tensor");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be 4D NZ tensors");
    TORCH_CHECK(q_out.dim() == 2 && q_out.scalar_type() == at::kBFloat16, "q_out must be 2D bfloat16");
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "k_cache/v_cache must be bfloat16 (this port does not quantize the KV cache)");
    TORCH_CHECK(num_head_q > 0 && num_head_k > 0 && num_head_v > 0, "head counts must be positive");
    TORCH_CHECK(num_head_k == num_head_v, "num_head_k and num_head_v must be equal");
    TORCH_CHECK(qkv.is_contiguous() && q_gamma.is_contiguous() && k_gamma.is_contiguous() &&
                    cos.is_contiguous() && sin.is_contiguous() && index.is_contiguous() &&
                    q_out.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(),
                "all tensors must be contiguous");

    const int64_t tokens = qkv.size(0);
    const int64_t dim = q_gamma.numel();
    TORCH_CHECK(dim == 128, "qkv_rms_norm_rope_cache only supports head dim 128");
    TORCH_CHECK(qkv.size(1) == (num_head_q + num_head_k + num_head_v) * dim, "qkv last dim mismatch");
    TORCH_CHECK(index.numel() == tokens, "index size must equal the token count");
    TORCH_CHECK(cos.size(0) == tokens && cos.size(1) == dim && sin.size(0) == tokens && sin.size(1) == dim,
                "cos/sin must be [tokens, D]");
    TORCH_CHECK(q_out.size(0) == tokens && q_out.size(1) == num_head_q * dim, "q_out must be [tokens, Nq*D]");
    TORCH_CHECK(k_cache.size(1) == num_head_k * dim / kBlockAlignNumFp16 && k_cache.size(3) == kBlockAlignNumFp16,
                "k_cache must be [blockNum, Nk*D/16, blockSize, 16]");
    TORCH_CHECK(v_cache.size(1) == num_head_v * dim / kBlockAlignNumFp16 && v_cache.size(3) == kBlockAlignNumFp16,
                "v_cache must be [blockNum, Nv*D/16, blockSize, 16]");
    TORCH_CHECK(k_cache.size(2) == v_cache.size(2), "k_cache and v_cache blockSize must match");
    TORCH_CHECK(k_cache.size(2) % kBlockAlignNumFp16 == 0, "blockSize must be a multiple of 16 elements");

    const auto tiling =
        BuildTiling(tokens, num_head_q, num_head_k, num_head_v, dim, k_cache.size(2), static_cast<float>(epsilon));
    const at::Tensor tiling_device = CopyTilingToDevice(tiling);
    const uint32_t block_dim = static_cast<uint32_t>(tiling.blockDim);
    EXEC_KERNEL_CMD(qkv_rms_norm_rope_cache, block_dim, qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache,
                    v_cache, tiling_device);

    return std::make_tuple(q_out, k_cache, v_cache);
}

}  // namespace npu_kernel
}  // namespace sglang
