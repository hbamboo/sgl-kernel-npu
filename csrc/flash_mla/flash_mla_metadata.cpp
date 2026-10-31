// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <dlfcn.h>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "flash_mla_host.h"

namespace sglang::npu_kernel {
using Optional=c10::optional<at::Tensor>;
namespace {
struct AclTensor;
struct Executor;
using Create=AclTensor*(*)(const int64_t*,uint64_t,aclDataType,const int64_t*,int64_t,aclFormat,const int64_t*,uint64_t,void*);
using Destroy=int(*)(const AclTensor*);
using Workspace=int(*)(const AclTensor*,const AclTensor*,const AclTensor*,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
    const char*,AclTensor*,uint64_t*,Executor**);
using Launch=int(*)(void*,uint64_t,Executor*,aclrtStream);
struct Api {
    Create create;Destroy destroy;Workspace workspace;Launch launch;
    // Different CANN vendors use the same libcust_opapi.so SONAME. Resolve
    // FlashMLA from configured vendor paths before a basename lookup can
    // return an unrelated library that the serving process already loaded.
    bool LoadCustom(const std::string& path) {
        auto custom=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if(!custom)return false;
        workspace=reinterpret_cast<Workspace>(dlsym(custom,"aclnnFlashMlaMetadataGetWorkspaceSize"));
        launch=reinterpret_cast<Launch>(dlsym(custom,"aclnnFlashMlaMetadata"));
        if(workspace && launch)return true;
        dlclose(custom);
        workspace=nullptr;launch=nullptr;
        return false;
    }
    Api() {
        auto base=dlopen("libnnopbase.so",RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(base,"FlashMLA metadata requires libnnopbase.so");
        create=reinterpret_cast<Create>(dlsym(base,"aclCreateTensor"));
        destroy=reinterpret_cast<Destroy>(dlsym(base,"aclDestroyTensor"));
        bool loaded=false;
        if(const char* roots=std::getenv("ASCEND_CUSTOM_OPP_PATH")) {
            std::istringstream paths(roots);
            std::string vendor;
            while(std::getline(paths,vendor,':')) {
                if(!vendor.empty() && LoadCustom(vendor+"/op_api/lib/libcust_opapi.so")) {
                    loaded=true;break;
                }
            }
        }
        if(!loaded)loaded=LoadCustom("libcust_opapi.so");
        TORCH_CHECK(create && destroy && loaded,
            "FlashMlaMetadata native ABI is unavailable. Set ASCEND_CUSTOM_OPP_PATH "
            "to the FlashMLA vendor directory containing op_api/lib/libcust_opapi.so");
    }
};
AclTensor* Descriptor(Api& api,const at::Tensor& t) {
    auto ptr=api.create(t.sizes().data(),t.dim(),ACL_INT32,t.strides().data(),0,ACL_FORMAT_ND,t.sizes().data(),t.dim(),t.data_ptr());
    TORCH_CHECK(ptr,"Cannot create FlashMLA metadata descriptor");return ptr;
}
}
at::Tensor flash_mla_metadata_impl(const at::Tensor& seq,int64_t heads,int64_t kvheads,
    const Optional& cu,const Optional& used,const c10::optional<int64_t>& maxq_,
    const c10::optional<int64_t>& maxkv_,int64_t dqk,int64_t dv,
    const c10::optional<int64_t>& mode_,int64_t left,int64_t right,
    const c10::optional<std::string>& layout_) {
    const c10::DeviceGuard guard(seq.device());
    auto maxq=maxq_.value_or(-1),maxkv=maxkv_.value_or(-1),mode=mode_.value_or(0);
    auto layout=layout_.value_or("BSND");
    TORCH_CHECK(seq.device().type()==c10::DeviceType::PrivateUse1 && seq.scalar_type()==at::kInt && seq.dim()==1 && seq.is_contiguous(),"cache_seqlens must be a contiguous NPU int32 vector");
    TORCH_CHECK(heads>0 && kvheads==1 && dqk==576 && dv==512,"FlashMLA requires Hkv=1,Dqk=576,Dv=512");
    const auto cores=FlashMlaCoreNums();
    for(const auto& t:{cu,used}) if(t)TORCH_CHECK(t->device()==seq.device() && t->scalar_type()==at::kInt && t->is_contiguous(),"metadata length tensors must be contiguous NPU int32");
    const int64_t words=(((cores.first+cores.second)*seq.numel()+1)*16+4095)/4096*4096;
    auto output=at::empty({words},seq.options());
    static Api api;
    std::vector<AclTensor*> tensors;
    auto descriptor=[&](const Optional& t){if(!t)return static_cast<AclTensor*>(nullptr);auto p=Descriptor(api,*t);tensors.push_back(p);return p;};
    auto cq=descriptor(cu),sq=descriptor(seq),uq=descriptor(used),out=descriptor(output);
    uint64_t bytes=0;Executor* executor=nullptr;
    auto rc=api.workspace(cq,sq,uq,maxq,maxkv,heads,kvheads,dqk,dv,mode,left,right,layout.c_str(),out,&bytes,&executor);
    if(rc!=0){for(auto t:tensors)api.destroy(t);TORCH_CHECK(false,"FlashMlaMetadataGetWorkspaceSize failed: ",rc);}
    auto scratch=at::empty({static_cast<int64_t>(bytes)},seq.options().dtype(at::kByte));
    auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto call=[scratch,executor,stream,tensors]() -> int {
        auto rc=api.launch(scratch.data_ptr(),scratch.numel(),executor,stream);
        for(auto t:tensors)api.destroy(t);
        TORCH_CHECK(rc==0,"FlashMlaMetadata launch failed: ",rc);return rc;
    };
    at_npu::native::OpCommand::RunOpApi("sgl_flash_mla_metadata",call);
    return output;
}
at::Tensor flash_mla_metadata(const at::Tensor& seq,int64_t heads,int64_t kvheads,
    const Optional& cu,const Optional& used,const c10::optional<int64_t>& maxq,
    const c10::optional<int64_t>& maxkv,int64_t dqk,int64_t dv,
    const c10::optional<int64_t>& mode,int64_t left,int64_t right,
    const c10::optional<std::string>& layout) {
    return flash_mla_metadata_impl(seq,heads,kvheads,cu,used,maxq,maxkv,dqk,dv,mode,left,right,layout);
}
}
TORCH_LIBRARY_FRAGMENT(npu,m) {
 m.def("flash_mla_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, int head_dim_qk=576, int head_dim_v=512, int? mask_mode=0, int win_left=-1, int win_right=-1, str? layout_q='BSND') -> Tensor");
}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("flash_mla_metadata",TORCH_FN(sglang::npu_kernel::flash_mla_metadata));}
