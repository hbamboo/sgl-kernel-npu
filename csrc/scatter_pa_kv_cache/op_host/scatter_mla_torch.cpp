// Licensed under the BSD 3-Clause License.
#include "acl/acl.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "scatter_mla_tiling.h"
#include "aclrtlaunch_sgl_scatter_mla.h"
#include <c10/core/DeviceGuard.h>
namespace sglang::npu_kernel {
void scatter_pa_kv_cache_mla(const at::Tensor& packed,const at::Tensor& rope,
                            const at::Tensor& slots,at::Tensor kc,at::Tensor vc,
                            int64_t nope_dim,int64_t tile_tokens,int64_t core_num,bool coalesce) {
    TORCH_CHECK(packed.device().type()==c10::DeviceType::PrivateUse1,"packed must be on NPU");
    c10::DeviceGuard guard(packed.device());
    TORCH_CHECK(packed.dim()==3&&rope.dim()==3&&rope.size(1)==1,"expected packed[T,H,nope+V], rope[T,1,R]");
    TORCH_CHECK(packed.scalar_type()==at::kBFloat16||packed.scalar_type()==at::kHalf,"MLA cache assembly supports BF16/FP16");
    TORCH_CHECK(packed.is_contiguous()&&rope.is_contiguous(),"packed and rope must be contiguous");
    TORCH_CHECK(slots.dim()==1&&slots.is_contiguous()&&(slots.scalar_type()==at::kLong||slots.scalar_type()==at::kInt),"slots must be contiguous int32/int64 [T]");
    TORCH_CHECK(kc.dim()==4&&vc.dim()==4&&kc.is_contiguous()&&vc.is_contiguous(),"caches must be contiguous [pages,page_size,H,D]");
    for(const auto& t:{rope,slots,kc,vc})TORCH_CHECK(t.device()==packed.device(),"all tensors must use the same NPU");
    for(const auto& t:{rope,kc,vc})TORCH_CHECK(t.scalar_type()==packed.scalar_type(),"all payload dtypes must match");
    const auto tokens=packed.size(0),heads=packed.size(1),rd=rope.size(2),vd=packed.size(2)-nope_dim;
    TORCH_CHECK(tokens==rope.size(0)&&slots.size(0)==tokens,"token counts must match");
    TORCH_CHECK(nope_dim>0&&rd>0&&vd>0&&nope_dim%16==0&&rd%16==0&&vd%16==0,"positive dimensions must be 32-byte aligned");
    TORCH_CHECK(kc.size(0)==vc.size(0)&&kc.size(1)==vc.size(1)&&kc.size(2)==heads&&vc.size(2)==heads&&kc.size(3)==nope_dim+rd&&vc.size(3)==vd,"cache geometry mismatch");
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t max_cores=platform->GetCoreNumAiv();
    TORCH_CHECK(core_num>=0&&core_num<=max_cores,"core_num exceeds device capacity");
    if(tokens==0)return;
    int64_t cores=core_num?core_num:max_cores;cores=std::min(cores,tokens);
    uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if(tile_tokens==0)tile_tokens=4;
    TORCH_CHECK(tile_tokens>0&&tile_tokens<=128,"tile_tokens out of range");
    ScatterMlaTiling p{};
    p.tokens=tokens;p.heads=heads;p.nopeDim=nope_dim;p.ropeDim=rd;p.valueDim=vd;
    p.capacity=kc.size(0)*kc.size(1);p.tileTokens=tile_tokens;p.usedCores=cores;
    p.tokensPerCore=(tokens+cores-1)/cores;p.indexBytes=slots.element_size();p.coalesce=coalesce;
    const auto required=2*tile_tokens*heads*((nope_dim+rd)+vd)*2+2*(tile_tokens*rd*2+256)+((p.tokensPerCore*p.indexBytes+31)/32)*32;
    TORCH_CHECK(required+2048<=static_cast<int64_t>(ub),"tile exceeds UB capacity: ",required," / ",ub);
    auto tiling=ge_helper::TilingTensorCache<ScatterMlaTiling>::Get(p,"scatter_pa_kv_cache_mla");
    const uint32_t block_dim=cores;
    EXEC_KERNEL_CMD(sgl_scatter_mla,block_dim,packed,rope,slots,kc,vc,tiling);
}
void scatter_pa_kv_cache_mla_meta(const at::Tensor&,const at::Tensor&,const at::Tensor&,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,bool) {}
}
TORCH_LIBRARY_FRAGMENT(npu,m) {
    m.def("scatter_pa_kv_cache_mla(Tensor packed, Tensor k_rope, Tensor slot_mapping, Tensor(a!) key_cache, Tensor(b!) value_cache, int nope_dim=192, int tile_tokens=0, int core_num=0, bool coalesce=True) -> ()");
}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("scatter_pa_kv_cache_mla",TORCH_FN(sglang::npu_kernel::scatter_pa_kv_cache_mla));}
TORCH_LIBRARY_IMPL(npu,Meta,m) {m.impl("scatter_pa_kv_cache_mla",TORCH_FN(sglang::npu_kernel::scatter_pa_kv_cache_mla_meta));}
