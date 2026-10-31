// Framework adapter for the unchanged upstream ScatterPaKvCache tiling math.
#pragma once
#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <ATen/ATen.h>

namespace sglang::scatter_port {
struct IntList {
    std::vector<int64_t> values;
    size_t GetSize() const { return values.size(); }
    const int64_t* GetData() const { return values.data(); }
};
struct Attrs {
    std::array<std::string,2> strings;
    std::array<IntList,2> lists;
    const char* GetStr(size_t i) const { return strings.at(i).c_str(); }
    const IntList* GetListInt(size_t i) const { return &lists.at(i-2); }
};
struct RawTiling {
    alignas(8) std::array<uint8_t,1024> data{};
    size_t bytes=0;
    void* GetData() { return data.data(); }
    size_t GetCapacity() const { return data.size(); }
    void SetDataSize(size_t n) { bytes=n; }
    size_t GetDataSize() const { return bytes; }
};
struct TensorInfo {
    gert::StorageShape shape;
    gert::CompileTimeTensorDesc desc;
    gert::Stride stride;
    std::unique_ptr<gert::Tensor> tensor;
    bool present=false;
    bool view=false;
};
class Context {
public:
    std::array<TensorInfo,8> inputs;
    Attrs attrs;
    RawTiling raw;
    uint32_t blockDim=0;
    uint64_t tilingKey=0;
    size_t workspace=0;
    const char* GetNodeName() const { return "sgl_kernel_npu::scatter_pa_kv_cache"; }
    const gert::CompileTimeTensorDesc* GetInputDesc(size_t i) const { return inputs.at(i).present ? &inputs.at(i).desc : nullptr; }
    const gert::CompileTimeTensorDesc* GetOptionalInputDesc(size_t i) const { return GetInputDesc(i); }
    const gert::StorageShape* GetInputShape(size_t i) const { return inputs.at(i).present ? &inputs.at(i).shape : nullptr; }
    const gert::Tensor* GetRequiredInputTensor(size_t i) const { return inputs.at(i).tensor.get(); }
    const gert::Tensor* GetOptionalInputTensor(size_t i) const { return inputs.at(i).tensor.get(); }
    bool InputIsView(size_t i) const { return inputs.at(i).view; }
    const gert::Stride* GetInputStride(size_t i) const { return &inputs.at(i).stride; }
    const Attrs* GetAttrs() const { return &attrs; }
    size_t* GetWorkspaceSizes(size_t) { return &workspace; }
    RawTiling* GetRawTilingData() { return &raw; }
    void SetBlockDim(uint32_t n) { blockDim=n; }
    void SetTilingKey(uint64_t n) { tilingKey=n; }
    uint64_t GetTilingKey() const { return tilingKey; }
    void Register(size_t i, const at::Tensor& t, ge::DataType dtype) {
        auto& s=inputs.at(i);
        s.present=t.defined();
        if (!s.present) return;
        gert::Shape shape;
        const bool packed_fp4=dtype==ge::DT_FLOAT4_E2M1||dtype==ge::DT_FLOAT4_E1M2;
        for (int64_t j=0;j<t.dim();++j) shape.AppendDim(t.size(j)*(packed_fp4&&j==t.dim()-1?2:1));
        s.shape.MutableOriginShape()=shape;
        s.shape.MutableStorageShape()=shape;
        s.desc.SetDataType(dtype);
        s.desc.SetOriginFormat(ge::FORMAT_ND);
        s.desc.SetStorageFormat(ge::FORMAT_ND);
        s.stride.SetDimNum(t.dim());
        for (int64_t j=0;j<t.dim();++j) s.stride.SetStride(j,t.stride(j)*(packed_fp4&&j<t.dim()-1?2:1));
        s.view=!t.is_contiguous();
        auto fmt=gert::StorageFormat(ge::FORMAT_ND,ge::FORMAT_ND,gert::ExpandDimsType());
        s.tensor=std::make_unique<gert::Tensor>(s.shape,fmt,dtype);
    }
};
// Only the upstream base-class execution sequence is needed by this operator.
class TilingBaseClass {
public:
    explicit TilingBaseClass(Context* c):context_(c) {}
    virtual ~TilingBaseClass()=default;
    ge::graphStatus DoTiling() {
        auto ret=GetShapeAttrsInfo(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        ret=GetPlatformInfo(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        if (!IsCapable()) return ge::GRAPH_PARAM_INVALID;
        ret=DoOpTiling(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        ret=DoLibApiTiling(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        ret=GetWorkspaceSize(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        ret=PostTiling(); if (ret!=ge::GRAPH_SUCCESS) return ret;
        context_->SetTilingKey(GetTilingKey());
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }
protected:
    Context* context_;
    uint64_t workspaceSize_=0;
    virtual bool IsCapable()=0;
    virtual ge::graphStatus GetShapeAttrsInfo()=0;
    virtual ge::graphStatus GetPlatformInfo()=0;
    virtual ge::graphStatus DoOpTiling()=0;
    virtual ge::graphStatus DoLibApiTiling()=0;
    virtual ge::graphStatus GetWorkspaceSize()=0;
    virtual ge::graphStatus PostTiling()=0;
    virtual uint64_t GetTilingKey() const=0;
    virtual void DumpTilingInfo() {}
};
}
namespace Ops::Base {
template<class T> inline T CeilDiv(T x,T y) { return (x+y-1)/y; }
template<class T> inline T FloorAlign(T x,T y) { return x/y*y; }
constexpr int64_t GetUbBlockSize() { return 32; }
constexpr int64_t GetUbBlockSize(sglang::scatter_port::Context*) { return 32; }
}
#define OP_CHECK_NULL_WITH_CONTEXT(c,p) do { TORCH_CHECK((p)!=nullptr,"scatter_pa_kv_cache: missing required metadata ",#p); } while(false)
#define OP_LOGE(...) ((void)0)
#define OP_LOGD(...) ((void)0)
#define OP_LOGI(...) ((void)0)
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(...) ((void)0)
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(...) ((void)0)
