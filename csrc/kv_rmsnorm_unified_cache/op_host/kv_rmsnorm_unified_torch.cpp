// Licensed under the BSD 3-Clause License.
//
// Host launcher + torch registration for kv_rmsnorm_unified_cache.
//
// This is a companion shared object (loaded by absolute path via
// torch.ops.load_library, like libsgl_kernel_npu_scatter_mla.so), so it does not
// touch the main sgl_kernel_npu extension registration.  The kernel is queued on
// the current NPU stream through OpCommand::RunOpApi, the same lifetime/stream
// convention used by scatter_pa_kv_cache_mla.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_sgl_kv_rmsnorm_unified_cache.h"

#include "kv_rmsnorm_unified_tiling.h"
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kCacheDim = KV_RMSNORM_CACHE_DIM;
constexpr int64_t kLatentDim = KV_RMSNORM_LATENT_DIM;
constexpr int64_t kRopeDim = KV_RMSNORM_ROPE_DIM;
constexpr int64_t kPageSize = KV_RMSNORM_PAGE_SIZE;
constexpr int64_t kUbReservedBytes = 1024;
// Real DAV_3510 contract (VECTOR_REG_WIDTH=256 B => 64 fp32 lanes), shared with
// the kernel header through kv_rmsnorm_unified_tiling.h.  The upstream RegBase
// RoPE helper stages two full vectors (2*VL_FP32 floats = 512 B).
static_assert(KV_RMSNORM_VREG_BYTES == 256 && KV_RMSNORM_VL_FP32 == 64,
              "host UB budget must match the installed CANN VECTOR_REG_WIDTH");
constexpr int64_t kRopeWspBytes = 2 * KV_RMSNORM_VL_FP32 * sizeof(float);
static_assert(kRopeWspBytes == 512, "RopeBasicComputeVF writes two 64-lane fp32 vectors");
static_assert(kRopeDim == 64 && kLatentDim == 512 && kCacheDim == 576 && kPageSize == 128,
              "coalesced scatter srcStride=0 requires the fixed aligned geometry gate");

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t CeilAlign(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

KvRmsNormUnifiedTiling BuildTiling(
    int64_t tokens, int64_t kvStride, int64_t cosStride, int64_t cacheRows, int64_t dtypeFlag, float epsilon)
{
    KvRmsNormUnifiedTiling p{};
    p.tokens = tokens;
    p.batchSize = tokens;
    p.seqLength = 1;
    p.numHead = 1;
    p.dk = kRopeDim;
    p.dv = kLatentDim;
    p.halfDk = kRopeDim / 2;
    p.dkAlign = CeilAlign(kRopeDim, 16);
    p.dvAlign = CeilAlign(kLatentDim, 16);
    p.halfDkAlign = CeilAlign(p.halfDk, 16);
    p.dkB8Align = CeilAlign(kRopeDim, 32);
    p.dvB8Align = CeilAlign(kLatentDim, 32);
    p.kvStride = kvStride;
    p.cosStride = cosStride;
    p.cacheDim = kCacheDim;
    p.cacheRows = cacheRows;
    p.epsilon = epsilon;
    p.reciprocal = 1.0f / static_cast<float>(kLatentDim);
    p.dtypeFlag = dtypeFlag;

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t ubSize = static_cast<int64_t>(ubBytes);
    const int64_t cores = std::max<int64_t>(1, static_cast<int64_t>(platform->GetCoreNumAiv()));

    const int64_t inUbSize = std::max(p.dkAlign, p.dvAlign) * 2;
    const int64_t inCosSinUbSize = 4 * p.halfDkAlign * 2;
    const int64_t gammaUbSize = p.dvAlign * 2;
    const int64_t outUbSize = p.dvAlign * 2;
    const int64_t ropeWspSize = kRopeWspBytes;
    const int64_t rmsNormWspSize = p.dvAlign * sizeof(float);
    p.inUbSize = inUbSize;
    p.outUbSize = outUbSize;
    p.rmsNormWspSize = rmsNormWspSize;

    int64_t ubFactor =
        (ubSize - kUbReservedBytes - gammaUbSize - ropeWspSize) /
        (2 * inUbSize + 2 * inCosSinUbSize + 2 * outUbSize + rmsNormWspSize);
    const int64_t blockFactor = CeilDiv(std::max<int64_t>(tokens, 1), cores);
    ubFactor = std::max<int64_t>(1, std::min(ubFactor, blockFactor));
    p.blockFactor = blockFactor;
    p.ubFactor = ubFactor;
    return p;
}

// P0 accepts only a fully packed per-layer 576 row space.  A layer view with a
// page gap or an interleaved multi-layer page envelope is rejected; the kernel
// addresses rows as slot*576 from the tensor data_ptr (which already includes
// the layer storage_offset).  numel() is not used as a row count because a
// strided view can cover more storage than it addresses.
int64_t CheckUnifiedCache(const at::Tensor& cache)
{
    TORCH_CHECK(cache.dim() == 3 || cache.dim() == 4, "cache must be [rows,1,576] or [pages,128,1,576]");
    TORCH_CHECK(cache.size(-1) == kCacheDim, "cache last dim must be 576");
    TORCH_CHECK(cache.stride(-1) == 1, "cache last dim must be contiguous");
    int64_t cacheRows = 0;
    if (cache.dim() == 4) {
        TORCH_CHECK(cache.size(1) == kPageSize && cache.size(2) == 1,
                    "unified PA cache must be [pages,128,1,576]");
        TORCH_CHECK(cache.stride(2) == kCacheDim && cache.stride(1) == kCacheDim,
                    "cache token rows must be packed at 576 elements");
        TORCH_CHECK(cache.stride(0) == kPageSize * kCacheDim,
                    "cache pages must be packed at 128*576 elements (no page gap)");
        cacheRows = cache.size(0) * kPageSize;
    } else {
        TORCH_CHECK(cache.size(1) == 1, "3D unified cache must be [rows,1,576]");
        TORCH_CHECK(cache.stride(0) == kCacheDim, "cache rows must be packed at 576 elements");
        cacheRows = cache.size(0);
    }
    TORCH_CHECK(cacheRows > 0, "cache must have at least one row");
    TORCH_CHECK(cache.numel() == cacheRows * kCacheDim, "unexpected cache element count");
    return cacheRows;
}

void CheckRowStridedKV(const at::Tensor& kv, int64_t tokens)
{
    TORCH_CHECK(kv.dim() == 4, "kv must be [T,1,1,576]");
    TORCH_CHECK(kv.size(0) == tokens && kv.size(1) == 1 && kv.size(2) == 1 && kv.size(3) == kCacheDim,
                "kv must be [T,1,1,576]");
    TORCH_CHECK(kv.stride(3) == 1, "kv last dim must be contiguous");
    TORCH_CHECK(kv.stride(0) > 0 && kv.stride(0) >= kCacheDim, "kv row stride must be positive and >= 576");
}

void CheckCosSin(const at::Tensor& t, const at::Tensor& kv, int64_t tokens, const char* name, int64_t& stride)
{
    TORCH_CHECK(t.dim() == 4 && t.size(0) == tokens && t.size(1) == 1 && t.size(2) == 1 && t.size(3) == kRopeDim,
                name, " must be [T,1,1,64]");
    TORCH_CHECK(t.scalar_type() == kv.scalar_type(), name, " dtype must match kv");
    TORCH_CHECK(t.device() == kv.device(), name, " device must match kv");
    TORCH_CHECK(t.stride(3) == 1 && t.stride(0) >= kRopeDim, name, " must have contiguous last dim and positive row stride");
    stride = t.stride(0);
}

void CheckOutput(const at::Tensor& out, const at::Tensor& kv, int64_t tokens, int64_t width, const char* name)
{
    TORCH_CHECK(out.dim() == 4 && out.size(0) == tokens && out.size(1) == 1 && out.size(2) == 1 &&
                    out.size(3) == width,
                name, " must be [T,1,1,", width, "]");
    TORCH_CHECK(out.scalar_type() == kv.scalar_type() && out.device() == kv.device(),
                name, " device/dtype must match kv");
    TORCH_CHECK(out.is_contiguous(), name, " must be contiguous");
}

} // namespace

void kv_rmsnorm_unified_cache(
    const at::Tensor& kv, const at::Tensor& gamma, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& index, at::Tensor cache, at::Tensor kRopeOut, at::Tensor cKvOut, double epsilon)
{
    TORCH_CHECK(kv.device().type() == c10::DeviceType::PrivateUse1, "kv must be an NPU tensor");
    c10::DeviceGuard guard(kv.device());
    const auto dtype = kv.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
                "kv_rmsnorm_unified_cache supports BF16/FP16 only (non-quantized)");
    for (const auto& t : {gamma, cos, sin, index, cache, kRopeOut, cKvOut}) {
        TORCH_CHECK(t.device() == kv.device(), "all tensors must use the same NPU device");
    }
    TORCH_CHECK(gamma.dim() == 1 && gamma.numel() == kLatentDim && gamma.is_contiguous(),
                "gamma must be a contiguous 512-element 1D tensor");
    TORCH_CHECK(gamma.scalar_type() == dtype, "gamma dtype must match kv");
    TORCH_CHECK(index.dim() == 1 && index.scalar_type() == at::kLong && index.is_contiguous(),
                "index must be a contiguous 1D int64 tensor");
    TORCH_CHECK(cache.scalar_type() == dtype, "cache dtype must match kv (no cache quantization on this route)");
    const int64_t cacheRows = CheckUnifiedCache(cache);

    const int64_t tokens = kv.size(0);
    CheckRowStridedKV(kv, tokens);
    TORCH_CHECK(index.numel() == tokens, "index size must equal the token count");
    int64_t cosStride = 0;
    int64_t sinStride = 0;
    CheckCosSin(cos, kv, tokens, "cos", cosStride);
    CheckCosSin(sin, kv, tokens, "sin", sinStride);
    TORCH_CHECK(cosStride == sinStride, "cos and sin row strides must match");
    CheckOutput(kRopeOut, kv, tokens, kRopeDim, "k_rope_out");
    CheckOutput(cKvOut, kv, tokens, kLatentDim, "c_kv_out");
    TORCH_CHECK(!kRopeOut.is_alias_of(cKvOut), "k_rope_out and c_kv_out must not alias each other");
    for (const auto& input : {kv, gamma, cos, sin, index, cache}) {
        TORCH_CHECK(!kRopeOut.is_alias_of(input), "k_rope_out must not alias inputs or the cache");
        TORCH_CHECK(!cKvOut.is_alias_of(input), "c_kv_out must not alias inputs or the cache");
    }
    // Writing the cache must never be able to overwrite an input that later rows
    // (or the returned outputs' consumers) still read.
    for (const auto& input : {kv, gamma, cos, sin, index}) {
        TORCH_CHECK(!cache.is_alias_of(input), "cache must not alias any input");
    }
    TORCH_CHECK(std::isfinite(epsilon), "epsilon must be finite");
    if (tokens == 0) {
        return;
    }

    const int64_t dtypeFlag = dtype == at::kBFloat16 ? 0 : 1;
    const auto tiling =
        BuildTiling(tokens, kv.stride(0), cosStride, cacheRows, dtypeFlag, static_cast<float>(epsilon));
    // The coalesced scatter's srcStride=0 is only valid while the UB row pitch
    // equals the copied width. Keep the fixed P0 geometry explicit here; any
    // generalization must update ScatterUnified's DMA parameters as well.
    TORCH_CHECK(tiling.dk == kRopeDim && tiling.dv == kLatentDim && tiling.dkAlign == tiling.dk &&
                    tiling.dvAlign == tiling.dv,
                "unsupported geometry for the packed coalesced scatter");

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t maxCores = static_cast<int64_t>(platform->GetCoreNumAiv());
    const int64_t usedCores = std::min<int64_t>(maxCores, CeilDiv(tokens, tiling.blockFactor));
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(1, usedCores));

    auto tilingTensor = ge_helper::TilingTensorCache<KvRmsNormUnifiedTiling>::Get(tiling, "kv_rmsnorm_unified_cache");
    auto npuStream = c10_npu::getCurrentNPUStream();
    auto aclStream = npuStream.stream(false);
    auto launch = [=]() -> int {
        for (const auto& t : {kv, gamma, cos, sin, index, cache, kRopeOut, cKvOut, tilingTensor}) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), npuStream);
        }
        return ACLRT_LAUNCH_KERNEL(sgl_kv_rmsnorm_unified_cache)(
            blockDim, aclStream, kv.data_ptr(), gamma.data_ptr(), cos.data_ptr(), sin.data_ptr(), index.data_ptr(),
            cache.data_ptr(), kRopeOut.data_ptr(), cKvOut.data_ptr(), tilingTensor.data_ptr());
    };
    at_npu::native::OpCommand::RunOpApi("kv_rmsnorm_unified_cache", launch);
}

void kv_rmsnorm_unified_cache_meta(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor,
    at::Tensor, at::Tensor, double)
{
}

} // namespace sglang::npu_kernel

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def(
        "kv_rmsnorm_unified_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index, "
        "Tensor(a!) cache, Tensor(b!) k_rope_out, Tensor(c!) c_kv_out, float epsilon=1e-5) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("kv_rmsnorm_unified_cache", TORCH_FN(sglang::npu_kernel::kv_rmsnorm_unified_cache));
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("kv_rmsnorm_unified_cache", TORCH_FN(sglang::npu_kernel::kv_rmsnorm_unified_cache_meta));
}
