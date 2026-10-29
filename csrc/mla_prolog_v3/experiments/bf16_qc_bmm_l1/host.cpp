#include "defines.h"
#include "torch_helper.h"
#include "aclrtlaunch_mla_prolog_v3_qc_bmm_l1.h"
namespace {
std::tuple<at::Tensor,at::Tensor> Fused(const at::Tensor& x,const at::Tensor& wqc,const at::Tensor& wuk,bool debug) {
    TORCH_CHECK(x.dim()==2 && x.size(1)==768 && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"requires contiguous BF16 [T,768]");
    TORCH_CHECK(wqc.sizes()==at::IntArrayRef({20,768,192}) && wqc.scalar_type()==at::kBFloat16 && wqc.is_contiguous(),"requires packed BF16 [20,768,192]");
    TORCH_CHECK(wuk.sizes()==at::IntArrayRef({20,192,512}) && wuk.scalar_type()==at::kBFloat16 && wuk.is_contiguous(),"requires BF16 [20,192,512]");
    TORCH_CHECK(x.size(0)>0 && x.device()==wqc.device() && x.device()==wuk.device(),"invalid token count or devices");
    auto out=at::empty({x.size(0),20,512},x.options());
    auto qc=debug?at::empty({x.size(0),20,192},x.options()):at::empty({0},x.options());
    uint32_t blockDim=32,tokens=x.size(0),emit=debug?1:0;void* unused=nullptr;
    EXEC_KERNEL_CMD(mla_prolog_v3_qc_bmm_l1,blockDim,x,wqc,wuk,out,qc,tokens,emit,unused,unused);
    return {out,qc};
}
TORCH_LIBRARY_FRAGMENT(npu,m) {m.def("mla_prolog_v3_qc_bmm_l1(Tensor x, Tensor wqc, Tensor wuk, bool debug=False) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("mla_prolog_v3_qc_bmm_l1",TORCH_FN(Fused));}
}
