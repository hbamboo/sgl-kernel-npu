// Licensed under the BSD 3-Clause License.
//
// Tiling blob shared by the host launcher and the AscendC kernel for
// kv_rmsnorm_rope_cache_unified576.  It is a plain POD (not a CANN
// BEGIN_TILING_DATA_DEF struct) so the direct-launch host can memcpy it into a
// cached device buffer and the kernel reads it back as a raw blob, following
// the existing qkv_rms_norm_rope_cache / scatter_mla direct-launch convention.
//
// The fields are a subset of the upstream KvRmsNormRopeCacheRegbaseFullLoadTilingData
// (ops-transformer 9.1.0, 6f0d3ec6). Quantization, NZ, block-table and
// non-PA cache fields were dropped together with those code paths.  kvStride,
// cosStride, cacheDim and cacheRows are the only semantic additions: they carry
// the real framework strides/geometry of the unified 576-wide latent cache.

#pragma once

#include <cstdint>

// Real DAV_3510 vector-register contract, taken from the installed CANN SDK
// (/usr/local/Ascend/cann-9.1.0/compiler/tikcpp/tikcfw/impl/utils/
// kernel_utils_constants.h: VECTOR_REG_WIDTH == 256 for every arch, including
// DAV3510).  The upstream platform::GetVRegSize() returns this same constant,
// so the RegBase kernel's VL_FP32 is 64 fp32 lanes.  Host and kernel share this
// header; the kernel header static_asserts platform::GetVRegSize() against it.
constexpr int64_t KV_RMSNORM_VREG_BYTES = 256;
constexpr int64_t KV_RMSNORM_VL_FP32 = KV_RMSNORM_VREG_BYTES / 4;

// Fixed P0 geometry.  The coalesced scatter sets srcStride=0 because the UB
// staging rows are packed exactly at the aligned copy width; that is only true
// for these dimensions (dkAlign == dk, dvAlign == dv).
constexpr int64_t KV_RMSNORM_ROPE_DIM = 64;
constexpr int64_t KV_RMSNORM_LATENT_DIM = 512;
constexpr int64_t KV_RMSNORM_CACHE_DIM = 576;
constexpr int64_t KV_RMSNORM_PAGE_SIZE = 128;

struct KvRmsNormUnifiedTiling {
    // Shape (token-major, N == 1).
    int64_t tokens;
    int64_t batchSize;   // == tokens; kept because the upstream row math uses it
    int64_t seqLength;   // == 1 (tokens are flattened across B*S)
    int64_t numHead;     // == 1

    // Latent / rope dimensions.
    int64_t dk;          // rope head dim, 64
    int64_t dv;          // latent (RMSNorm) dim, 512
    int64_t halfDk;      // dk / 2, 32
    int64_t dkAlign;     // 32-byte aligned dk
    int64_t dvAlign;     // 32-byte aligned dv
    int64_t halfDkAlign; // 32-byte aligned halfDk
    int64_t dkB8Align;   // retained layout field, unused by the b16 path
    int64_t dvB8Align;   // retained layout field, unused by the b16 path

    // Real input/cache geometry.
    int64_t kvStride;    // elements between consecutive kv rows in GM (>= dk+dv)
    int64_t cosStride;   // elements between consecutive cos/sin rows in GM (>= dk)
    int64_t cacheDim;    // 576: unified cache row width
    int64_t cacheRows;   // rows addressable in the unified cache view

    // Work split.
    int64_t blockFactor; // rows per core
    int64_t ubFactor;    // rows per UB tile
    int64_t inUbSize;    // bytes per row for the kv staging queue
    int64_t outUbSize;   // bytes per row for the output staging queue
    int64_t rmsNormWspSize; // bytes per row for the RMSNorm reduce workspace

    // Numeric parameters.
    float epsilon;
    float reciprocal;

    // 0 = bfloat16, 1 = float16.
    int64_t dtypeFlag;
};
