// Isolated three-launch ABI. No tiling H2D copy and no production headers.
#include "acl/acl.h"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_moe_local_ranks.h"
#include "aclrtlaunch_moe_prefix_offsets.h"
#include "aclrtlaunch_moe_scatter_payload.h"

namespace moe_stable_proof {
using at::Tensor;
using Outputs = std::tuple<Tensor, Tensor, Tensor, Tensor>;
void Validate(const Tensor& x, const Tensor& ids, const Tensor& scale, int64_t tile) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float8_e4m3fn && x.dim() == 2 &&
                x.size(0) >= 8192 && x.size(0) <= 36000 && x.size(1) == 2048,
                "requires FP8 E4M3 [T in 8192..36000,2048]");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 &&
                ids.size(0) == x.size(0) && ids.size(1) == 4, "requires int32 IDs[T,4]");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float8_e8m0fnu && scale.dim() == 3 &&
                scale.size(0) == x.size(0) && scale.size(1) == 32 && scale.size(2) == 2,
                "requires E8M0 scale[T,32,2]");
    TORCH_CHECK(ids.device() == x.device() && scale.device() == x.device() &&
                x.is_contiguous() && ids.is_contiguous() && scale.is_contiguous(),
                "all inputs must be contiguous on the same device");
    TORCH_CHECK(tile > 0 && tile <= 16, "tile_rows must be in 1..16");
}
Outputs Allocate(const Tensor& x) {
    auto length = x.size(0)*4;
    return {at::empty({length,2048}, x.options()), at::empty({length}, x.options().dtype(at::kInt)),
            at::empty({64}, x.options().dtype(at::kLong)),
            at::empty({length,32,2}, x.options().dtype(at::kByte))};
}
Outputs Route(const Tensor& x, const Tensor& ids, const Tensor& scale, int64_t tile) {
    Validate(x, ids, scale, tile);
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "requires NPU input");
    c10::DeviceGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream(); auto raw = stream.stream(false);
    // No CPU inspection of IDs. Upstream top-k must guarantee every ID in [0,64).
    auto output = Allocate(x);
    auto out = std::get<0>(output), map = std::get<1>(output);
    auto totals = std::get<2>(output), outScale = std::get<3>(output);
    auto ranks = at::empty({x.size(0)*4}, ids.options());
    auto counts = at::empty({64,64}, ids.options());
    auto offsets = at::empty({64,64}, ids.options());
    uint32_t tokens = x.size(0), perCore = (tokens+63)/64;
    uint32_t sortSlots = ((perCore*4+31)/32)*32, stride = 1;
    while (stride < tokens*4) stride *= 2;
    auto launch = [=]() -> int {
        // External inputs may belong to another allocation stream.
        // All seven fresh tensors above belong to raw and remain captured
        // until this callback has submitted their last same-stream use.
        for (const auto& t : {x, ids, scale})
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        auto status = ACLRT_LAUNCH_KERNEL(moe_local_ranks)(64, raw, ids.data_ptr(), ranks.data_ptr(),
            counts.data_ptr(), tokens, perCore, sortSlots, stride);
        if (status != ACL_ERROR_NONE) return status;
        status = ACLRT_LAUNCH_KERNEL(moe_prefix_offsets)(64, raw, counts.data_ptr(),
            offsets.data_ptr(), totals.data_ptr());
        if (status != ACL_ERROR_NONE) return status;
        return ACLRT_LAUNCH_KERNEL(moe_scatter_payload)(64, raw, x.data_ptr(), scale.data_ptr(),
            ids.data_ptr(), ranks.data_ptr(), offsets.data_ptr(), out.data_ptr(), outScale.data_ptr(),
            map.data_ptr(), tokens, perCore, sortSlots, static_cast<uint32_t>(tile));
    };
    at_npu::native::OpCommand::RunOpApi("moe_stable_three_kernel", launch);
    return output;
}
Outputs Meta(const Tensor& x, const Tensor& ids, const Tensor& scale, int64_t tile) {
    Validate(x, ids, scale, tile); return Allocate(x);
}
}
TORCH_LIBRARY(moe_stable_proof, m) {
    m.def("route_fp8(Tensor x, Tensor ids, Tensor scale, int tile_rows=8) -> (Tensor,Tensor,Tensor,Tensor)");
}
TORCH_LIBRARY_IMPL(moe_stable_proof, PrivateUse1, m) {
    m.impl("route_fp8", TORCH_FN(moe_stable_proof::Route));
}
TORCH_LIBRARY_IMPL(moe_stable_proof, Meta, m) {
    m.impl("route_fp8", TORCH_FN(moe_stable_proof::Meta));
}
