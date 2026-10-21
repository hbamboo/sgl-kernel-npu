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
// AscendC entry for qkv_rms_norm_rope_cache.
//
// The Ascend 950 (DAV_3510) build uses the RegBase kernel (register-resident norm+RoPE,
// one UB read + one UB write per head).  See qkv_rms_norm_rope_cache_regbase.h.

#include "kernel_operator.h"

#include "../op_host/qkv_rms_norm_rope_cache_tiling.h"

using namespace sglang::npu_kernel;

#include "qkv_rms_norm_rope_cache_regbase.h"

namespace {
__aicore__ inline void CopyTilingData(QkvRmsNormRopeCacheTilingData *dst, GM_ADDR src)
{
    uint32_t *const dst32b = reinterpret_cast<uint32_t *>(dst);
    const __gm__ uint32_t *const src32b = reinterpret_cast<__gm__ uint32_t *>(src);
    for (uint32_t i = 0; i < sizeof(QkvRmsNormRopeCacheTilingData) / sizeof(uint32_t); ++i) {
        dst32b[i] = src32b[i];
    }
}
}  // namespace

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR q_out,
    GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    QkvRmsNormRopeCacheTilingData tilingData;
    CopyTilingData(&tilingData, tiling);
    KernelQkvRmsNormRopeCacheRegbase op(&pipe, &tilingData);
    op.Init(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache);
    op.Process();
}
