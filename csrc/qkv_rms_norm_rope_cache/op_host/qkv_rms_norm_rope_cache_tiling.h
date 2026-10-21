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
// Tiling blob shared by the host launcher and the RegBase kernel.
//
// The original CANN op derives this struct from BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF.
// Here it is a plain POD so both sides agree byte-for-byte; the host memcpy's it into a
// device buffer and the kernel copies it back as a raw blob.
//
// Only fields the current kernel actually reads are kept (the upstream fields for quant,
// ND outputs, workspace and the UB-tile kernel were dropped together with those code paths).

#pragma once

#include <cstdint>

namespace sglang {
namespace npu_kernel {

struct QkvRmsNormRopeCacheTilingData {
    int64_t batchSize;    // tokens (S is folded into B, seqLength is always 1)
    int64_t seqLength;    // kept for the token-index math
    int64_t numHead;      // N = Nq + Nk + Nv
    int64_t qkvDim;       // head dim D (128)
    int64_t numHeadQ;     // Nq
    int64_t numHeadK;     // Nk
    int64_t numHeadV;     // Nv
    int64_t blockSize;    // page size of k_cache / v_cache
    float epsilon;        // RMSNorm epsilon

    int64_t blockFactorQ;  // heads handled by one Q core
    int64_t blockFactorK;  // heads handled by one K core
    int64_t blockFactorV;  // heads handled by one V core
    int64_t blockDimQ;     // number of Q cores
    int64_t blockDimK;     // number of K cores
    int64_t blockDimV;     // number of V cores
    int64_t blockDim;      // total launched cores = blockDimQ + blockDimK + blockDimV
    int64_t ubFactor;      // heads staged per UB tile
};

}  // namespace npu_kernel
}  // namespace sglang
