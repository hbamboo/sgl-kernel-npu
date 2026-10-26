// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// mla_prolog_v3 host side: builds the tiling context that the ops-transformer
// tiling implementation expects, runs it, and reports the kernel launch
// parameters.  The tiling code itself is reused unmodified from
// csrc/mla_prolog/op_host (see the port notes in README-port.md).

#ifndef MLA_PROLOG_V3_HOST_H
#define MLA_PROLOG_V3_HOST_H

#include <cstdint>
#include <string>
#include <vector>

#include "graph/types.h"
#include "../op_kernel/mla_prolog_tiling_data.h"

namespace sglang {
namespace npu_kernel {

// Plain description of every tensor of MlaPrologV3.  An empty `dims` vector
// means "optional input not provided".
struct MlaPrologV3Tensors {
    // ---- required inputs
    std::vector<int64_t> tokenX;     // (T, He) or (B, S, He)
    std::vector<int64_t> weightDq;   // (He, Hcq)   FRACTAL_NZ
    std::vector<int64_t> weightUqQr; // (Hcq, N * (D + Dr)) FRACTAL_NZ
    std::vector<int64_t> weightUk;   // (N, D, Hckv)
    std::vector<int64_t> weightDkvKr; // (He, Hckv + Dr) FRACTAL_NZ
    std::vector<int64_t> gammaCq;    // (Hcq,)
    std::vector<int64_t> gammaCkv;   // (Hckv,)
    std::vector<int64_t> ropeSin;    // (T, Dr)
    std::vector<int64_t> ropeCos;    // (T, Dr)
    std::vector<int64_t> kvCache;    // (BlockNum, BlockSize, Nkv, Hckv)
    std::vector<int64_t> krCache;    // (BlockNum, BlockSize, Nkv, Dr)

    // ---- optional inputs
    std::vector<int64_t> cacheIndex;
    std::vector<int64_t> dequantScaleX;
    std::vector<int64_t> dequantScaleWDq;
    std::vector<int64_t> dequantScaleWUqQr;
    std::vector<int64_t> dequantScaleWDkvKr;
    std::vector<int64_t> quantScaleCkv;
    std::vector<int64_t> quantScaleCkr;
    std::vector<int64_t> smoothScalesCq;
    std::vector<int64_t> actualSeqLen;
    std::vector<int64_t> kNopeClipAlpha;

    // ---- outputs (shapes are needed by the tiling checks)
    std::vector<int64_t> query;
    std::vector<int64_t> queryRope;
    std::vector<int64_t> kvCacheOut;
    std::vector<int64_t> krCacheOut;
    std::vector<int64_t> dequantScaleQNope;
    std::vector<int64_t> queryNorm;
    std::vector<int64_t> dequantScaleQNorm;

    // ---- dtypes
    ge::DataType tokenXType{ge::DT_BF16};
    ge::DataType weightDqType{ge::DT_BF16};
    ge::DataType weightUqQrType{ge::DT_BF16};
    ge::DataType weightUkType{ge::DT_BF16};
    ge::DataType weightDkvKrType{ge::DT_BF16};
    ge::DataType gammaCqType{ge::DT_BF16};
    ge::DataType gammaCkvType{ge::DT_BF16};
    ge::DataType ropeSinType{ge::DT_BF16};
    ge::DataType ropeCosType{ge::DT_BF16};
    ge::DataType kvCacheType{ge::DT_BF16};
    ge::DataType krCacheType{ge::DT_BF16};
    ge::DataType cacheIndexType{ge::DT_INT64};
    ge::DataType scaleType{ge::DT_FLOAT};
    ge::DataType scaleXType{ge::DT_FLOAT};
    ge::DataType scaleWDqType{ge::DT_FLOAT};
    ge::DataType scaleWUqQrType{ge::DT_FLOAT};
    ge::DataType scaleWDkvKrType{ge::DT_FLOAT};
    ge::DataType scaleQNopeType{ge::DT_FLOAT};
    ge::DataType scaleQNormType{ge::DT_FLOAT};
    ge::DataType queryNormType{ge::DT_BF16};
    ge::DataType actualSeqLenType{ge::DT_INT32};
    ge::DataType clipAlphaType{ge::DT_FLOAT};
    ge::DataType queryType{ge::DT_BF16};
    ge::DataType queryRopeType{ge::DT_BF16};

    // ---- attributes
    float rmsEpsCq{1e-5f};
    float rmsEpsCkv{1e-5f};
    std::string cacheMode{"PA_BSND"};
    bool queryNormFlag{true};
    int64_t weightQuantMode{0};
    int64_t kvQuantMode{0};
    int64_t queryQuantMode{0};
    int64_t ckvkrRepoMode{0};
    int64_t quantScaleRepoMode{0};
    int64_t tileSize{128};
    float qcQrScale{1.0f};
    float kcScale{1.0f};
    bool doRope{true};
};

struct MlaPrologV3TilingResult {
    uint32_t blockDim{0};
    size_t workspaceSize{0};
    uint64_t tilingKey{0};
    // decoded tiling-key template arguments (mirrored into MlaPrologTilingData)
    uint32_t cacheMode{0};
    uint32_t scenario{0};
    uint32_t quantMode{0};
    uint32_t enableDequantOpt{0};
    uint32_t enableGroupComputeOpt{0};
    uint32_t emptyTensorMode{0};
    uint32_t actualSeqLenMode{0};
    uint32_t splitMMode{0};
    uint32_t enableRope{0};
    uint32_t cvMode{0};
};

// Runs the (unmodified) ops-transformer tiling for MlaPrologV3.
// Returns false and fills `err` when the tiling rejects the configuration.
bool MlaPrologV3RunTiling(const MlaPrologV3Tensors &tensors, optiling::MlaPrologTilingData *tilingData,
                          MlaPrologV3TilingResult *result, std::string *err);

} // namespace npu_kernel
} // namespace sglang

#endif // MLA_PROLOG_V3_HOST_H
