// Licensed under the BSD 3-Clause License.
// Source-compatible AscendC ScatterPaKvCache entry in sgl-kernel-npu conventions.
#include "acl/acl.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "scatter_pa_kv_cache_tiling.h"
#include "aclrtlaunch_sgl_scatter_pa_kv_cache.h"
#include "aclrtlaunch_sgl_scatter_pa_compression_4.h"
#include "aclrtlaunch_sgl_scatter_pa_compression_5.h"
#include "aclrtlaunch_sgl_scatter_pa_compression_7.h"
#include "aclrtlaunch_sgl_scatter_pa_compression_20.h"
#include "aclrtlaunch_sgl_scatter_pa_compression_21.h"
#include "aclrtlaunch_sgl_scatter_pa_mixed_nz.h"
#include <c10/core/DeviceGuard.h>
#include <tuple>

namespace sglang::npu_kernel {
namespace {
ge::DataType GeType(at::ScalarType t) {
    switch(t) {
        case at::kBFloat16:return ge::DT_BF16;
        case at::kHalf:return ge::DT_FLOAT16;
        case at::kFloat:return ge::DT_FLOAT;
        case at::kChar:return ge::DT_INT8;
        case at::kByte:return ge::DT_UINT8;
        case at::kShort:return ge::DT_INT16;
        case at::kInt:return ge::DT_INT32;
        case at::kLong:return ge::DT_INT64;
        case at::kUInt16:return ge::DT_UINT16;
        case at::kUInt32:return ge::DT_UINT32;
        case at::kFloat4_e2m1fn_x2:return ge::DT_FLOAT4_E2M1;
        case at::kFloat8_e4m3fn:return ge::DT_FLOAT8_E4M3FN;
        case at::kFloat8_e5m2:return ge::DT_FLOAT8_E5M2;
        default:TORCH_CHECK(false,"scatter_pa_kv_cache: unsupported dtype ",t);
    }
}
uint64_t KernelType(at::ScalarType t,bool arithmetic=false) {
    if(arithmetic) {
        if(t==at::kByte)return 4;
        if(t==at::kUInt16)return 5;
        if(t==at::kUInt32)return 7;
        if(t==at::kFloat8_e4m3fn)return 20;
        if(t==at::kFloat8_e5m2)return 21;
    }
    if(t==at::kBFloat16)return 12;
    if(t==at::kHalf)return 1;
    if(t==at::kFloat)return 0;
    if(at::elementSize(t)==1)return 2;
    if(at::elementSize(t)==2)return 6;
    if(at::elementSize(t)==4)return 3;
    TORCH_CHECK(false,"scatter_pa_kv_cache: unsupported payload element size");
}
}

void scatter_pa_kv_cache(const at::Tensor& key,at::Tensor key_cache,const at::Tensor& slots,
                        const at::Tensor& value,at::Tensor value_cache,
                        const c10::optional<at::Tensor>& compress_lens,
                        const c10::optional<at::Tensor>& compress_seq_offset,
                        const c10::optional<at::Tensor>& seq_lens,
                        const std::string& cache_mode,const std::string& scatter_mode,
                        at::IntArrayRef strides,at::IntArrayRef offsets) {
    TORCH_CHECK(key.device().type()==c10::DeviceType::PrivateUse1,"key must be an NPU tensor");
    c10::DeviceGuard guard(key.device());
    TORCH_CHECK(key.dim()==3||key.dim()==4,"key must be rank 3 or 4");
    TORCH_CHECK(value.dim()==key.dim(),"key/value ranks must match");
    TORCH_CHECK(key_cache.dim()==4||key_cache.dim()==5,"key_cache must be rank 4 or 5");
    TORCH_CHECK(value_cache.dim()==key_cache.dim(),"cache ranks must match");
    TORCH_CHECK(slots.scalar_type()==at::kInt||slots.scalar_type()==at::kLong,"slots must be int32 or int64");
    TORCH_CHECK(slots.is_contiguous(),"slot_mapping must be contiguous");
    TORCH_CHECK(strides.size()==2&&offsets.size()==2,"strides/offsets must contain two values");
    TORCH_CHECK(key.scalar_type()==key_cache.scalar_type()&&value.scalar_type()==value_cache.scalar_type(),"each cache dtype must match its payload");
    TORCH_CHECK(cache_mode=="PA_NZ"||key.scalar_type()==value.scalar_type(),"K/V dtypes must match outside PA_NZ");
    for(const auto& t:{key_cache,slots,value,value_cache})TORCH_CHECK(t.device()==key.device(),"all tensors must be on the same NPU");
    for(const auto& t:{compress_lens,compress_seq_offset,seq_lens}) if(t) {
        TORCH_CHECK(t->device()==key.device(),"optional tensors must be on the same NPU");
        TORCH_CHECK(t->scalar_type()==slots.scalar_type(),"optional index dtypes must match slot_mapping");
        TORCH_CHECK(t->is_contiguous(),"optional index tensors must be contiguous");
    }
    if(key.numel()==0&&value.numel()==0)return;
    scatter_port::Context ctx;
    const auto dtype=GeType(key.scalar_type());
    ctx.Register(0,key,dtype);ctx.Register(1,key_cache,dtype);ctx.Register(2,slots,GeType(slots.scalar_type()));
    ctx.Register(3,value,GeType(value.scalar_type()));ctx.Register(4,value_cache,GeType(value.scalar_type()));
    ctx.Register(5,compress_lens.value_or(at::Tensor()),GeType(slots.scalar_type()));
    ctx.Register(6,compress_seq_offset.value_or(at::Tensor()),GeType(slots.scalar_type()));
    ctx.Register(7,seq_lens.value_or(at::Tensor()),GeType(slots.scalar_type()));
    ctx.attrs.strings={cache_mode,scatter_mode};
    ctx.attrs.lists[0].values=strides.vec();ctx.attrs.lists[1].values=offsets.vec();
    optiling::ScatterPaKvCacheTiling tiler(&ctx,2);
    TORCH_CHECK(tiler.DoTiling()==ge::GRAPH_SUCCESS,"scatter_pa_kv_cache: invalid shape, layout or attributes");
    TORCH_CHECK(key.scalar_type()!=at::kFloat4_e2m1fn_x2||ctx.tilingKey<200||ctx.tilingKey>=500,"upstream FP4 supports byte-copy Norm/NZ/Nct paths only");
    ScatterPaKvCacheLaunchData launch{};
    TORCH_CHECK(ctx.raw.bytes==sizeof(launch.data),"unexpected upstream tiling ABI");
    std::memcpy(&launch.data,ctx.raw.data.data(),sizeof(launch.data));
    const bool arithmetic=ctx.tilingKey==240||ctx.tilingKey==241||ctx.tilingKey==280||ctx.tilingKey==281||ctx.tilingKey==400||ctx.tilingKey==401;
    const bool mixed=key.scalar_type()!=value.scalar_type();
    launch.key=ctx.tilingKey;launch.dtype=mixed ? key.element_size()|(uint64_t(value.element_size())<<32) : KernelType(key.scalar_type(),arithmetic);launch.indexBytes=slots.element_size();
    auto tiling=ge_helper::TilingTensorCache<ScatterPaKvCacheLaunchData>::Get(launch,"scatter_pa_kv_cache");
    void* p0=key.data_ptr();void* p1=key_cache.data_ptr();void* p2=slots.data_ptr();
    void* p3=value.data_ptr();void* p4=value_cache.data_ptr();
    void* p5=compress_lens?compress_lens->data_ptr():nullptr;
    void* p6=compress_seq_offset?compress_seq_offset->data_ptr():nullptr;
    void* p7=seq_lens?seq_lens->data_ptr():nullptr;
    const uint32_t block_dim=ctx.blockDim;
    if(mixed) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_mixed_nz,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else if(arithmetic&&launch.dtype==4) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_compression_4,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else if(arithmetic&&launch.dtype==5) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_compression_5,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else if(arithmetic&&launch.dtype==7) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_compression_7,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else if(arithmetic&&launch.dtype==20) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_compression_20,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else if(arithmetic&&launch.dtype==21) {
        EXEC_KERNEL_CMD(sgl_scatter_pa_compression_21,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    } else {
        EXEC_KERNEL_CMD(sgl_scatter_pa_kv_cache,block_dim,p0,p1,p2,p3,p4,p5,p6,p7,tiling);
    }
}
void scatter_pa_kv_cache_meta(const at::Tensor&,at::Tensor,const at::Tensor&,
                             const at::Tensor&,at::Tensor,const c10::optional<at::Tensor>&,
                             const c10::optional<at::Tensor>&,const c10::optional<at::Tensor>&,
                             const std::string&,const std::string&,at::IntArrayRef,at::IntArrayRef) {}
}

TORCH_LIBRARY_FRAGMENT(npu,m) {
    m.def("scatter_pa_kv_cache(Tensor key, Tensor(a!) key_cache, Tensor slot_mapping, Tensor value, Tensor(b!) value_cache, Tensor? compress_lens=None, Tensor? compress_seq_offset=None, Tensor? seq_lens=None, str cache_mode='Norm', str scatter_mode='None', int[] strides=[1,1], int[] offsets=[0,0]) -> ()");
}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) { m.impl("scatter_pa_kv_cache",TORCH_FN(sglang::npu_kernel::scatter_pa_kv_cache)); }
TORCH_LIBRARY_IMPL(npu,Meta,m) { m.impl("scatter_pa_kv_cache",TORCH_FN(sglang::npu_kernel::scatter_pa_kv_cache_meta)); }
