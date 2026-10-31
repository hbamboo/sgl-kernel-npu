#pragma once
#include <ATen/ATen.h>
#include <string>
#include <acl/acl.h>
#include <map>
#include <mutex>
namespace sglang::npu_kernel {
inline std::pair<int64_t,int64_t> FlashMlaCoreNums() {
    static std::mutex mutex;
    static std::map<int,std::pair<int64_t,int64_t>> values;
    int device=0;TORCH_CHECK(aclrtGetDevice(&device)==0,"Cannot query NPU device");
    std::lock_guard<std::mutex> lock(mutex);
    auto it=values.find(device);if(it!=values.end())return it->second;
    int64_t cube=0,vector=0;
    TORCH_CHECK(aclrtGetDeviceInfo(device,ACL_DEV_ATTR_CUBE_CORE_NUM,&cube)==0 && aclrtGetDeviceInfo(device,ACL_DEV_ATTR_VECTOR_CORE_NUM,&vector)==0 && cube>0 && vector==cube*2,"FlashMLA requires a 1:2 Cube/Vector platform");
    return values.emplace(device,std::make_pair(cube,vector)).first->second;
}
at::Tensor flash_mla_metadata_impl(const at::Tensor&,int64_t,int64_t,
    const c10::optional<at::Tensor>&,const c10::optional<at::Tensor>&,
    const c10::optional<int64_t>&,const c10::optional<int64_t>&,int64_t,int64_t,
    const c10::optional<int64_t>&,int64_t,int64_t,const c10::optional<std::string>&);
}
