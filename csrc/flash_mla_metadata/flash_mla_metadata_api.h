// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
//
// Shared FlashMLA core/API surface used by both the FlashMLA compute companion
// and the independently built metadata companion. It contains only platform
// core detection and the host metadata entry point declaration; the metadata
// algorithm and launch ABI live in csrc/flash_mla_metadata.
#pragma once
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace sglang::npu_kernel {
inline std::pair<int64_t, int64_t> FlashMlaCoreNums()
{
    static std::mutex mutex;
    static std::map<int, std::pair<int64_t, int64_t>> values;
    int device = 0;
    TORCH_CHECK(aclrtGetDevice(&device) == 0, "Cannot query NPU device");
    std::lock_guard<std::mutex> lock(mutex);
    auto it = values.find(device);
    if (it != values.end()) {
        return it->second;
    }
    int64_t cube = 0;
    int64_t vector = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cube) == 0 &&
                    aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector) == 0 &&
                    cube > 0 && vector == cube * 2,
                "FlashMLA requires a 1:2 Cube/Vector platform");
    return values.emplace(device, std::make_pair(cube, vector)).first->second;
}

at::Tensor flash_mla_metadata_impl(const at::Tensor&, int64_t, int64_t,
    const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
    const c10::optional<int64_t>&, const c10::optional<int64_t>&, int64_t, int64_t,
    const c10::optional<int64_t>&, int64_t, int64_t, const c10::optional<std::string>&);
} // namespace sglang::npu_kernel
