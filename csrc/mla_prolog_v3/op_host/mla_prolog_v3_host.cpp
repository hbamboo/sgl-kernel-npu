// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// mla_prolog_v3 host side: builds the ops-transformer tiling context from plain
// tensor descriptions, runs the (unmodified) tiling and returns the launch
// parameters.  Tensor pointers are not touched here; only shapes/dtypes/attrs.

#include "mla_prolog_v3_host.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

#include "mla_prolog_tiling.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr const char *kOpName = "MlaPrologV3";

struct TensorSlot {
    gert::StorageShape shape;
    gert::CompileTimeTensorDesc desc;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    switch (dims.size()) {
        case 0U:
            return gert::StorageShape({}, {});
        case 1U:
            return gert::StorageShape({dims[0]}, {dims[0]});
        case 2U:
            return gert::StorageShape({dims[0], dims[1]}, {dims[0], dims[1]});
        case 3U:
            return gert::StorageShape({dims[0], dims[1], dims[2]}, {dims[0], dims[1], dims[2]});
        case 4U:
            return gert::StorageShape({dims[0], dims[1], dims[2], dims[3]},
                                      {dims[0], dims[1], dims[2], dims[3]});
        case 5U:
            return gert::StorageShape({dims[0], dims[1], dims[2], dims[3], dims[4]},
                                      {dims[0], dims[1], dims[2], dims[3], dims[4]});
        default:
            throw std::runtime_error("[mla_prolog_v3] unsupported tensor rank");
    }
}

// Fills a context parameter entry.  `present == false` marks an optional input
// that the caller did not provide (desc/shape stay nullptr).
void Fill(const gert::CompileTimeTensorDesc *&descOut, const gert::StorageShape *&shapeOut, TensorSlot &slot,
          const std::vector<int64_t> &dims, ge::DataType dtype, bool isNz = false, bool present = true)
{
    if (!present || dims.empty()) {
        descOut = nullptr;
        shapeOut = nullptr;
        return;
    }
    slot.shape = MakeStorageShape(dims);
    slot.desc.SetDataType(dtype);
    slot.desc.SetOriginFormat(isNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND);
    slot.desc.SetStorageFormat(isNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND);
    descOut = &slot.desc;
    shapeOut = &slot.shape;
}

} // namespace

bool MlaPrologV3RunTiling(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                          MlaPrologV3TilingResult *result, std::string *err)
{
    // Stable storage for the gert objects referenced by the context.
    std::deque<TensorSlot> slots;
    auto nextSlot = [&slots]() -> TensorSlot & {
        slots.emplace_back();
        return slots.back();
    };

    optiling::MlaPrologContext ctx{};
    ctx.opName = kOpName;
    ctx.opType = kOpName;

    TensorSlot &sTokenX = nextSlot();
    TensorSlot &sWDq = nextSlot();
    TensorSlot &sWUqQr = nextSlot();
    TensorSlot &sWUk = nextSlot();
    TensorSlot &sWDkvKr = nextSlot();
    TensorSlot &sGammaCq = nextSlot();
    TensorSlot &sGammaCkv = nextSlot();
    TensorSlot &sRopeSin = nextSlot();
    TensorSlot &sRopeCos = nextSlot();
    TensorSlot &sKvCache = nextSlot();
    TensorSlot &sKrCache = nextSlot();
    TensorSlot &sCacheIdx = nextSlot();
    TensorSlot &sScaleX = nextSlot();
    TensorSlot &sScaleWDq = nextSlot();
    TensorSlot &sScaleWUqQr = nextSlot();
    TensorSlot &sScaleWDkvKr = nextSlot();
    TensorSlot &sQuantCkv = nextSlot();
    TensorSlot &sQuantCkr = nextSlot();
    TensorSlot &sSmoothCq = nextSlot();
    TensorSlot &sActualSeq = nextSlot();
    TensorSlot &sClipAlpha = nextSlot();
    TensorSlot &sQuery = nextSlot();
    TensorSlot &sQueryRope = nextSlot();
    TensorSlot &sKvCacheOut = nextSlot();
    TensorSlot &sKrCacheOut = nextSlot();
    TensorSlot &sScaleQNope = nextSlot();
    TensorSlot &sQueryNorm = nextSlot();
    TensorSlot &sScaleQNorm = nextSlot();

    Fill(ctx.tokenX.desc, ctx.tokenX.shape, sTokenX, tensors.tokenX, tensors.tokenXType);
    Fill(ctx.weightDq.desc, ctx.weightDq.shape, sWDq, tensors.weightDq, tensors.weightDqType, true);
    Fill(ctx.weightUqQr.desc, ctx.weightUqQr.shape, sWUqQr, tensors.weightUqQr, tensors.weightUqQrType, true);
    Fill(ctx.weightUk.desc, ctx.weightUk.shape, sWUk, tensors.weightUk, tensors.weightUkType);
    Fill(ctx.weightDkvKr.desc, ctx.weightDkvKr.shape, sWDkvKr, tensors.weightDkvKr, tensors.weightDkvKrType, true);
    Fill(ctx.rmsnormGammaCq.desc, ctx.rmsnormGammaCq.shape, sGammaCq, tensors.gammaCq, tensors.gammaCqType);
    Fill(ctx.rmsnormGammaCkv.desc, ctx.rmsnormGammaCkv.shape, sGammaCkv, tensors.gammaCkv, tensors.gammaCkvType);
    Fill(ctx.ropeSin.desc, ctx.ropeSin.shape, sRopeSin, tensors.ropeSin, tensors.ropeSinType);
    Fill(ctx.ropeCos.desc, ctx.ropeCos.shape, sRopeCos, tensors.ropeCos, tensors.ropeCosType);
    Fill(ctx.kvCache.desc, ctx.kvCache.shape, sKvCache, tensors.kvCache, tensors.kvCacheType);
    Fill(ctx.krCache.desc, ctx.krCache.shape, sKrCache, tensors.krCache, tensors.krCacheType);

    Fill(ctx.cacheIndex.desc, ctx.cacheIndex.shape, sCacheIdx, tensors.cacheIndex, tensors.cacheIndexType);
    Fill(ctx.dequantScaleX.desc, ctx.dequantScaleX.shape, sScaleX, tensors.dequantScaleX, tensors.scaleXType);
    Fill(ctx.dequantScaleWDq.desc, ctx.dequantScaleWDq.shape, sScaleWDq, tensors.dequantScaleWDq,
         tensors.scaleWDqType);
    Fill(ctx.dequantScaleWUqQr.desc, ctx.dequantScaleWUqQr.shape, sScaleWUqQr, tensors.dequantScaleWUqQr,
         tensors.scaleWUqQrType);
    Fill(ctx.dequantScaleWDkvKr.desc, ctx.dequantScaleWDkvKr.shape, sScaleWDkvKr, tensors.dequantScaleWDkvKr,
         tensors.scaleWDkvKrType);
    Fill(ctx.quantScaleCkv.desc, ctx.quantScaleCkv.shape, sQuantCkv, tensors.quantScaleCkv, tensors.scaleType);
    Fill(ctx.quantScaleCkr.desc, ctx.quantScaleCkr.shape, sQuantCkr, tensors.quantScaleCkr, tensors.scaleType);
    Fill(ctx.smoothScalesCq.desc, ctx.smoothScalesCq.shape, sSmoothCq, tensors.smoothScalesCq, tensors.scaleType);
    Fill(ctx.actualSeqLen.desc, ctx.actualSeqLen.shape, sActualSeq, tensors.actualSeqLen, tensors.actualSeqLenType);
    Fill(ctx.kNopeClipAlpha.desc, ctx.kNopeClipAlpha.shape, sClipAlpha, tensors.kNopeClipAlpha,
         tensors.clipAlphaType);

    Fill(ctx.query.desc, ctx.query.shape, sQuery, tensors.query, tensors.queryType);
    Fill(ctx.queryRope.desc, ctx.queryRope.shape, sQueryRope, tensors.queryRope, tensors.queryRopeType);
    Fill(ctx.kvCacheOut.desc, ctx.kvCacheOut.shape, sKvCacheOut, tensors.kvCacheOut, tensors.kvCacheType);
    Fill(ctx.krCacheOut.desc, ctx.krCacheOut.shape, sKrCacheOut, tensors.krCacheOut, tensors.krCacheType);
    Fill(ctx.dequantScaleQNope.desc, ctx.dequantScaleQNope.shape, sScaleQNope, tensors.dequantScaleQNope,
         tensors.scaleQNopeType);
    Fill(ctx.queryNorm.desc, ctx.queryNorm.shape, sQueryNorm, tensors.queryNorm, tensors.queryNormType);
    Fill(ctx.dequantScaleQNorm.desc, ctx.dequantScaleQNorm.shape, sScaleQNorm, tensors.dequantScaleQNorm,
         tensors.scaleQNormType);

    // Attributes live on the stack; the context only keeps pointers.
    float epsCq = tensors.rmsEpsCq;
    float epsCkv = tensors.rmsEpsCkv;
    std::string cacheMode = tensors.cacheMode;
    bool queryNormFlag = tensors.queryNormFlag;
    int64_t weightQuantMode = tensors.weightQuantMode;
    int64_t kvQuantMode = tensors.kvQuantMode;
    int64_t queryQuantMode = tensors.queryQuantMode;
    int64_t ckvkrRepoMode = tensors.ckvkrRepoMode;
    int64_t quantScaleRepoMode = tensors.quantScaleRepoMode;
    int64_t tileSize = tensors.tileSize;
    float qcQrScale = tensors.qcQrScale;
    float kcScale = tensors.kcScale;
    bool doRope = tensors.doRope;

    ctx.rmsNormEspilonCq = &epsCq;
    ctx.rmsNormEspilonCkv = &epsCkv;
    ctx.cacheMode = cacheMode.c_str();
    ctx.queryNormFlag = &queryNormFlag;
    ctx.weightQuantMode = &weightQuantMode;
    ctx.kvQuantMode = &kvQuantMode;
    ctx.queryQuantMode = &queryQuantMode;
    ctx.ckvkrRepoMode = &ckvkrRepoMode;
    ctx.quantScaleRepoMode = &quantScaleRepoMode;
    ctx.tileSize = &tileSize;
    ctx.qcQrScale = &qcQrScale;
    ctx.kcScale = &kcScale;
    ctx.doRope = &doRope;
    ctx.doRopeValue = doRope;

    // [sgl-kernel-npu port] PA cache stride of dim 0 in elements, i.e. what
    // ConvertContext would read from the GE tensor descriptor.  The kernel uses it to
    // address PA blocks, so a zero stride silently breaks every token that spills into
    // block >= 1.
    auto stride0 = [](const std::vector<int64_t> &dims) -> uint64_t {
        if (dims.empty()) {
            return 0U;
        }
        uint64_t s = 1U;
        for (size_t i = 1U; i < dims.size(); ++i) {
            s *= static_cast<uint64_t>(dims[i]);
        }
        return s;
    };
    ctx.kvCacheStride0 = stride0(tensors.kvCache);
    ctx.krCacheStride0 = stride0(tensors.krCache);

    size_t workSpaces[2] = {0U, 0U};
    ctx.workSpaces = workSpaces;

    optiling::MlaPrologTiling tiling;
    const ge::graphStatus st = tiling.RunBigKernelTiling(ctx, tilingData);
    if (st != ge::GRAPH_SUCCESS && std::getenv("DSH_MLA_V3_DEBUG") != nullptr) {
        auto dump = [](const char *name, const std::vector<int64_t> &d) {
            std::fprintf(stderr, "[dsh-v3] %-22s", name);
            for (int64_t v : d) {
                std::fprintf(stderr, " %ld", static_cast<long>(v));
            }
            std::fprintf(stderr, "\n");
        };
        dump("tokenX", tensors.tokenX);
        dump("weightDq", tensors.weightDq);
        dump("weightUqQr", tensors.weightUqQr);
        dump("weightUk", tensors.weightUk);
        dump("weightDkvKr", tensors.weightDkvKr);
        dump("gammaCq", tensors.gammaCq);
        dump("gammaCkv", tensors.gammaCkv);
        dump("ropeSin", tensors.ropeSin);
        dump("kvCache", tensors.kvCache);
        dump("krCache", tensors.krCache);
        dump("query", tensors.query);
        dump("queryRope", tensors.queryRope);
        dump("queryNorm", tensors.queryNorm);
        dump("dequantScaleQNope", tensors.dequantScaleQNope);
        dump("dequantScaleQNorm", tensors.dequantScaleQNorm);
        dump("cacheIndex", tensors.cacheIndex);
        std::fprintf(stderr, "[dsh-v3] types: x=%d dq=%d uqqr=%d uk=%d dkvkr=%d qn=%d eps=(%f,%f) mode=%d kvq=%d qnf=%d\n",
                     static_cast<int>(tensors.tokenXType), static_cast<int>(tensors.weightDqType),
                     static_cast<int>(tensors.weightUqQrType), static_cast<int>(tensors.weightUkType),
                     static_cast<int>(tensors.weightDkvKrType), static_cast<int>(tensors.queryNormType),
                     tensors.rmsEpsCq, tensors.rmsEpsCkv, static_cast<int>(tensors.weightQuantMode),
                     static_cast<int>(tensors.kvQuantMode), static_cast<int>(tensors.queryNormFlag));
    }
    if (st != ge::GRAPH_SUCCESS) {
        if (err != nullptr) {
            *err = "[mla_prolog_v3] RunBigKernelTiling failed";
        }
        return false;
    }

    if (result != nullptr) {
        result->blockDim = ctx.blockDim;
        result->workspaceSize = workSpaces[0];
        result->tilingKey = ctx.tilingKey;
        result->cacheMode = ctx.tplArgs.cacheMode;
        result->scenario = ctx.tplArgs.scenario;
        result->quantMode = ctx.tplArgs.quantMode;
        result->enableDequantOpt = ctx.tplArgs.enableDequantOpt;
        result->enableGroupComputeOpt = ctx.tplArgs.enableGroupComputeOpt;
        result->emptyTensorMode = ctx.tplArgs.emptyTensorMode;
        result->actualSeqLenMode = ctx.tplArgs.actualSeqLenMode;
        result->splitMMode = ctx.tplArgs.splitMMode;
        result->enableRope = ctx.tplArgs.enableRope;
        result->cvMode = ctx.tplArgs.cvMode;
    }
    tilingData->tplArgs = ctx.tplArgs;
    tilingData->tilingKey = ctx.tilingKey;
    return true;
}


namespace {

// Allocation-free FNV-1a over the full (shape, dtype, attr) configuration.
inline void HashBytes(uint64_t *h, const void *data, size_t bytes)
{
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        *h ^= static_cast<uint64_t>(p[i]);
        *h *= 1099511628211ULL;
    }
}

inline void HashDims(uint64_t *h, const std::vector<int64_t> &dims)
{
    const int64_t n = static_cast<int64_t>(dims.size());
    HashBytes(h, &n, sizeof(n));
    if (!dims.empty()) {
        HashBytes(h, dims.data(), dims.size() * sizeof(int64_t));
    }
}

inline void HashValue(uint64_t *h, int64_t v)
{
    HashBytes(h, &v, sizeof(v));
}

uint64_t MakeTilingHash(const MlaPrologV3Tensors &t)
{
    uint64_t h = 1469598103934665603ULL;
    HashDims(&h, t.tokenX);
    HashDims(&h, t.weightDq);
    HashDims(&h, t.weightUqQr);
    HashDims(&h, t.weightUk);
    HashDims(&h, t.weightDkvKr);
    HashDims(&h, t.gammaCq);
    HashDims(&h, t.gammaCkv);
    HashDims(&h, t.ropeSin);
    HashDims(&h, t.ropeCos);
    HashDims(&h, t.kvCache);
    HashDims(&h, t.krCache);
    HashDims(&h, t.cacheIndex);
    HashDims(&h, t.dequantScaleX);
    HashDims(&h, t.dequantScaleWDq);
    HashDims(&h, t.dequantScaleWUqQr);
    HashDims(&h, t.dequantScaleWDkvKr);
    HashDims(&h, t.quantScaleCkv);
    HashDims(&h, t.quantScaleCkr);
    HashDims(&h, t.smoothScalesCq);
    HashDims(&h, t.actualSeqLen);
    HashDims(&h, t.kNopeClipAlpha);
    HashDims(&h, t.query);
    HashDims(&h, t.queryRope);
    HashDims(&h, t.queryNorm);
    HashDims(&h, t.dequantScaleQNorm);
    const int dtypes[] = {static_cast<int>(t.tokenXType),      static_cast<int>(t.weightDqType),
                          static_cast<int>(t.weightUqQrType),  static_cast<int>(t.weightUkType),
                          static_cast<int>(t.weightDkvKrType), static_cast<int>(t.gammaCqType),
                          static_cast<int>(t.gammaCkvType),    static_cast<int>(t.ropeSinType),
                          static_cast<int>(t.ropeCosType),     static_cast<int>(t.kvCacheType),
                          static_cast<int>(t.krCacheType),     static_cast<int>(t.scaleXType),
                          static_cast<int>(t.scaleWDqType),    static_cast<int>(t.scaleWUqQrType),
                          static_cast<int>(t.scaleWDkvKrType), static_cast<int>(t.scaleQNopeType),
                          static_cast<int>(t.scaleQNormType),  static_cast<int>(t.queryNormType)};
    HashBytes(&h, dtypes, sizeof(dtypes));
    HashBytes(&h, t.cacheMode.data(), t.cacheMode.size());
    const double attrs[] = {t.rmsEpsCq,     t.rmsEpsCkv,      static_cast<double>(t.weightQuantMode),
                            static_cast<double>(t.kvQuantMode), static_cast<double>(t.queryQuantMode),
                            static_cast<double>(t.ckvkrRepoMode), static_cast<double>(t.quantScaleRepoMode),
                            static_cast<double>(t.tileSize), static_cast<double>(t.qcQrScale),
                            static_cast<double>(t.kcScale), static_cast<double>(t.queryNormFlag),
                            static_cast<double>(t.doRope)};
    HashBytes(&h, attrs, sizeof(attrs));
    return h;
}

} // namespace

bool MlaPrologV3RunTilingCached(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                                MlaPrologV3TilingResult *result, std::string *err)
{
    struct Entry {
        optiling::MlaPrologTilingData tilingData;
        MlaPrologV3TilingResult result;
        bool ok;
        std::string err;
    };
    static std::mutex mutex;
    static std::unordered_map<uint64_t, Entry> cache;

    const uint64_t key = MakeTilingHash(tensors);
    {
        std::lock_guard<std::mutex> guard(mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            *tilingData = it->second.tilingData;
            *result = it->second.result;
            if (!it->second.ok && err != nullptr) {
                *err = it->second.err;
            }
            return it->second.ok;
        }
    }

    Entry entry{};
    entry.ok = MlaPrologV3RunTiling(tensors, &entry.tilingData, &entry.result, &entry.err);

    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.emplace(key, entry).first;
    *tilingData = it->second.tilingData;
    *result = it->second.result;
    if (!it->second.ok && err != nullptr) {
        *err = it->second.err;
    }
    return it->second.ok;
}

} // namespace npu_kernel
} // namespace sglang
