#pragma once
#include "scatter_pa_kv_cache_port_context.h"
#include "scatter_pa_kv_cache_tiling_data.h"
namespace optiling {
struct ScatterPaKvCacheHostTilingData : ::ScatterPaKvCacheTilingData {
    void set_usedCoreNum(int64_t v) { usedCoreNum=v; }
    int64_t get_usedCoreNum() const { return usedCoreNum; }
    void set_blockFactor(int64_t v) { blockFactor=v; }
    int64_t get_blockFactor() const { return blockFactor; }
    void set_tailBlockFactor(int64_t v) { tailBlockFactor=v; }
    int64_t get_tailBlockFactor() const { return tailBlockFactor; }
    void set_kHandleNumPerCore(int64_t v) { kHandleNumPerCore=v; }
    int64_t get_kHandleNumPerCore() const { return kHandleNumPerCore; }
    void set_vHandleNumPerCore(int64_t v) { vHandleNumPerCore=v; }
    int64_t get_vHandleNumPerCore() const { return vHandleNumPerCore; }
    void set_kTailHandleNum(int64_t v) { kTailHandleNum=v; }
    int64_t get_kTailHandleNum() const { return kTailHandleNum; }
    void set_vTailHandleNum(int64_t v) { vTailHandleNum=v; }
    int64_t get_vTailHandleNum() const { return vTailHandleNum; }
    void set_kLoopNum(int64_t v) { kLoopNum=v; }
    int64_t get_kLoopNum() const { return kLoopNum; }
    void set_vLoopNum(int64_t v) { vLoopNum=v; }
    int64_t get_vLoopNum() const { return vLoopNum; }
    void set_kHandleNumPerLoop(int64_t v) { kHandleNumPerLoop=v; }
    int64_t get_kHandleNumPerLoop() const { return kHandleNumPerLoop; }
    void set_vHandleNumPerLoop(int64_t v) { vHandleNumPerLoop=v; }
    int64_t get_vHandleNumPerLoop() const { return vHandleNumPerLoop; }
    void set_keyStride0(int64_t v) { keyStride0=v; }
    int64_t get_keyStride0() const { return keyStride0; }
    void set_keyStride1(int64_t v) { keyStride1=v; }
    int64_t get_keyStride1() const { return keyStride1; }
    void set_keyStride2(int64_t v) { keyStride2=v; }
    int64_t get_keyStride2() const { return keyStride2; }
    void set_valueStride0(int64_t v) { valueStride0=v; }
    int64_t get_valueStride0() const { return valueStride0; }
    void set_valueStride1(int64_t v) { valueStride1=v; }
    int64_t get_valueStride1() const { return valueStride1; }
    void set_valueStride2(int64_t v) { valueStride2=v; }
    int64_t get_valueStride2() const { return valueStride2; }
    void set_kHeadSize(int64_t v) { kHeadSize=v; }
    int64_t get_kHeadSize() const { return kHeadSize; }
    void set_vHeadSize(int64_t v) { vHeadSize=v; }
    int64_t get_vHeadSize() const { return vHeadSize; }
    void set_batch(int64_t v) { batch=v; }
    int64_t get_batch() const { return batch; }
    void set_numBlocks(int64_t v) { numBlocks=v; }
    int64_t get_numBlocks() const { return numBlocks; }
    void set_blockSize(int64_t v) { blockSize=v; }
    int64_t get_blockSize() const { return blockSize; }
    void set_seqLen(int64_t v) { seqLen=v; }
    int64_t get_seqLen() const { return seqLen; }
    void set_numHead(int64_t v) { numHead=v; }
    int64_t get_numHead() const { return numHead; }
    void set_numTokens(int64_t v) { numTokens=v; }
    int64_t get_numTokens() const { return numTokens; }
    void set_ubSize(int64_t v) { ubSize=v; }
    int64_t get_ubSize() const { return ubSize; }
    void set_kStride(int64_t v) { kStride=v; }
    int64_t get_kStride() const { return kStride; }
    void set_vStride(int64_t v) { vStride=v; }
    int64_t get_vStride() const { return vStride; }
    void set_kOffset(int64_t v) { kOffset=v; }
    int64_t get_kOffset() const { return kOffset; }
    void set_vOffset(int64_t v) { vOffset=v; }
    int64_t get_vOffset() const { return vOffset; }
    void set_keyCacheStride0(int64_t v) { keyCacheStride0=v; }
    int64_t get_keyCacheStride0() const { return keyCacheStride0; }
    void set_keyCacheStride1(int64_t v) { keyCacheStride1=v; }
    int64_t get_keyCacheStride1() const { return keyCacheStride1; }
    void set_keyCacheStride2(int64_t v) { keyCacheStride2=v; }
    int64_t get_keyCacheStride2() const { return keyCacheStride2; }
    void set_keyCacheStride3(int64_t v) { keyCacheStride3=v; }
    int64_t get_keyCacheStride3() const { return keyCacheStride3; }
    void set_valueCacheStride0(int64_t v) { valueCacheStride0=v; }
    int64_t get_valueCacheStride0() const { return valueCacheStride0; }
    void set_valueCacheStride1(int64_t v) { valueCacheStride1=v; }
    int64_t get_valueCacheStride1() const { return valueCacheStride1; }
    void set_valueCacheStride2(int64_t v) { valueCacheStride2=v; }
    int64_t get_valueCacheStride2() const { return valueCacheStride2; }
    void set_valueCacheStride3(int64_t v) { valueCacheStride3=v; }
    int64_t get_valueCacheStride3() const { return valueCacheStride3; }
    void set_kCacheBlockStride(int64_t v) { kCacheBlockStride=v; }
    int64_t get_kCacheBlockStride() const { return kCacheBlockStride; }
    void set_vCacheBlockStride(int64_t v) { vCacheBlockStride=v; }
    int64_t get_vCacheBlockStride() const { return vCacheBlockStride; }
    size_t GetDataSize() const { return sizeof(::ScatterPaKvCacheTilingData); }
    void SaveToBuffer(void* p,size_t cap) const { TORCH_CHECK(cap>=GetDataSize(),"tiling buffer too small"); std::memcpy(p,static_cast<const ::ScatterPaKvCacheTilingData*>(this),GetDataSize()); }
};
}
namespace optiling {
struct ScatterPaKvCacheCompileInfo {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
};

class ScatterPaKvCacheTiling : public sglang::scatter_port::TilingBaseClass {
public:
    explicit ScatterPaKvCacheTiling(sglang::scatter_port::Context *context, int64_t inOutMode)
        : sglang::scatter_port::TilingBaseClass(context), inOutMode_(inOutMode)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus TemplateNormal();
    ge::graphStatus TemplateRope();
    ge::graphStatus TemplateAlibi();
    ge::graphStatus TemplateOmni();
    ge::graphStatus TemplateNZ();
    ge::graphStatus GetIndexDtype();
    ge::graphStatus GetInputDtype();
    ge::graphStatus CheckDimValid();
    ge::graphStatus CheckNormal();
    ge::graphStatus CheckRope();
    ge::graphStatus CheckAlibi();
    ge::graphStatus CheckNz();
    ge::graphStatus GetTemplateType(int64_t inputKeyDimNum);
    ge::graphStatus GetContiguousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx);
    void GetNonContigousStrideInfo();
    ge::graphStatus GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx);
    void SetInputPos();
    void GetCommonTilingInfo();
    ge::graphStatus CheckSlotMappingShape(int64_t requiredDimNum);
    int64_t RoundUp(int64_t x, int64_t dtypeSize);
    void GenTilingKey();
    bool IsNonContiguous();
    bool IsAxesContiguous(const gert::Stride &stride, const gert::Shape &shape, int64_t startAxis, int64_t endAxis);
    bool IsLastAxisContiguous(const gert::Stride &stride, const gert::Shape &shape);

private:
    platform_ascendc::SocVersion socVersion_;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_{};
    int64_t tailBlockFactor_ = 0;
    int64_t kHandleNumPerCore_ = 0;
    int64_t vHandleNumPerCore_ = 0;
    int64_t kLoopNum_ = 0;
    int64_t vLoopNum_ = 0;
    int64_t kHandleNumPerLoop_ = 0;
    int64_t vHandleNumPerLoop_ = 0;
    int64_t kTailHandleNum_ = 0;
    int64_t vTailHandleNum_ = 0;
    int64_t numTokens_ = 0;
    int64_t dtypeByteSize_ = 0;
    int64_t valueDtypeByteSize_ = 0;
    int64_t keyStride0_ = 0;
    int64_t keyStride1_ = 0;
    int64_t keyStride2_ = 0;
    int64_t valueStride0_ = 0;
    int64_t valueStride1_ = 0;
    int64_t valueStride2_ = 0;
    int64_t keyCacheStride0_ = 0;
    int64_t keyCacheStride1_ = 0;
    int64_t keyCacheStride2_ = 0;
    int64_t keyCacheStride3_ = 0;
    int64_t valueCacheStride0_ = 0;
    int64_t valueCacheStride1_ = 0;
    int64_t valueCacheStride2_ = 0;
    int64_t valueCacheStride3_ = 0;
    int64_t kHeadSize_{};
    int64_t vHeadSize_{};
    int64_t batch_{};
    int64_t numHead_{};
    int64_t numBlocks_{};
    int64_t blockSize_{};
    int64_t seqLen_{};
    uint64_t tilingKey_{};
    int64_t templateType_ = 0;   // 1: normal, 2: rope, 3: alibi
    int64_t indexDtypeSize_ = 0; // int32 -> 4, int64 -> 8
    int64_t isFullyLoad_ = 0;    // 0: not fully load, 1: fully load
    int64_t inOutMode_{};          // 1: single input and output，2: dual input and output
    int64_t inputKey_{};
    int64_t inputKeyCacheIn_{};
    int64_t inputSlotMapping_{};
    int64_t inputValue_{};
    int64_t inputValueCacheIn_{};
    int64_t inputCompressLens_{};
    int64_t inputCompressSeqOffset_{};
    int64_t inputSeqLens_{};
    int64_t kStride_ = 0;
    int64_t vStride_ = 0;
    int64_t kOffset_ = 0;
    int64_t vOffset_ = 0;

    gert::Shape inputKeyShape_;
    gert::Shape inputKeyCacheInShape_;
    gert::Shape slotMappingShape_;
    gert::Shape inputValueShape_;
    gert::Shape inputValueCacheInShape_;
    gert::Shape compressLensShape_;
    gert::Shape compressSeqOffsetShape_;
    gert::Shape seqLensShape_;

    gert::Stride keyStride_;
    gert::Stride valueStride_;
    gert::Stride keyCacheStride_;
    gert::Stride valueCacheStride_;

    ge::DataType inputDtype_;
    ge::DataType valueDtype_;
    ScatterPaKvCacheHostTilingData tilingData_{};
};

struct ScatterPaKvCacheMembaseParams {
    platform_ascendc::SocVersion socVersion;
    int64_t numTokens{0};
    int64_t numHead{0};
    int64_t kHeadSize{0};
    int64_t vHeadSize{0};
    int64_t blockSize{0};
    int64_t typeByteK{0};
    int64_t typeByteV{0};
    int64_t typeByteSlot{0};
    int64_t numBatchs{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t kStride{0};
    int64_t vStride{0};
    int64_t kOffset{0};
    int64_t vOffset{0};
    int64_t kCacheBlockStride{0};
    int64_t vCacheBlockStride{0};

    int64_t tilingKey{0};
    uint64_t workspaceSize{0};

    uint64_t sysWorkspaceSize{0};
    int64_t usedCoreNum{0};
    uint64_t ubSize{0};

    int64_t templateType{0}; // 1: normal, 2: nz, 3: alibi, 4: rope, 5:siso, 6:omni, 7:normal_nct, 8:siso_nct
};

class ScatterPaKvCacheMembaseTiling : public sglang::scatter_port::TilingBaseClass {
public:
    explicit ScatterPaKvCacheMembaseTiling(sglang::scatter_port::Context *context) : sglang::scatter_port::TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus GetInputDtypeSiso();
    ge::graphStatus GetInputDtypeKv();
    ge::graphStatus GetInputDtype();
    ge::graphStatus CheckInputDimNumSiso();
    ge::graphStatus CheckInputDimNumNorm();
    ge::graphStatus CheckInputDimNumNz();
    ge::graphStatus CheckInputDimNumCompress();
    ge::graphStatus CheckInputDimNumNHSD();
    ge::graphStatus CheckInputDimNum();
    ge::graphStatus CheckInputShapeSiso();
    ge::graphStatus CheckInputShapeNorm();
    ge::graphStatus CheckInputShapeNz();
    ge::graphStatus CheckInputShapeCompress();
    ge::graphStatus CheckInputShape();
    ge::graphStatus GetTemplateType();
    ge::graphStatus DoNoCompressOpTiling();
    ge::graphStatus DoCompressAlibiOpTiling();
    ge::graphStatus DoCompressRopeAndOmniOpTiling();
    ge::graphStatus CheckInputShapeNHSD();
    ge::graphStatus DoNHSDOpTiling();
private:
    ScatterPaKvCacheMembaseParams params_;
    ScatterPaKvCacheHostTilingData tilingData_{};
    gert::Shape inputKeyShape_;
    gert::Shape inputKeyCacheInShape_;
    gert::Shape slotMappingShape_;
    gert::Shape inputValueShape_;
    gert::Shape inputValueCacheInShape_;
    gert::Shape compressLensShape_;
    gert::Shape compressSeqOffsetShape_;
    gert::Shape seqLensShape_;
    ge::DataType inputKeyDtype_;
    ge::DataType inputValueDtype_;
};

} // namespace optiling
