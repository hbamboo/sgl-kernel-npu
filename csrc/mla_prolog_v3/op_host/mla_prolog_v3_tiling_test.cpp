// Copyright (c) 2025 Huawei Technologies Co., Ltd
// Standalone tiling driver for MlaPrologV3 (not part of the extension build).
//
//   ./mla_prolog_v3_tiling_test <He> <Hcq> <Hckv> <D> <Dr> <N> <T> <BlockNum> <BlockSize> <wqMode> <kvqMode>
//
// Prints the tiling key, blockDim, workspace and the decoded template arguments.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mla_prolog_v3_host.h"

using sglang::npu_kernel::MlaPrologV3Tensors;
using sglang::npu_kernel::MlaPrologV3TilingResult;
using sglang::npu_kernel::MlaPrologV3RunTiling;

int main(int argc, char **argv)
{
    auto arg = [&](int idx, long def) -> long { return argc > idx ? std::atol(argv[idx]) : def; };
    const int64_t He = arg(1, 7168);
    const int64_t Hcq = arg(2, 1536);
    const int64_t Hckv = arg(3, 512);
    const int64_t D = arg(4, 128);
    const int64_t Dr = arg(5, 64);
    const int64_t N = arg(6, 32);
    const int64_t T = arg(7, 8);
    const int64_t blockNum = arg(8, 2);
    const int64_t blockSize = arg(9, 128);
    const int64_t wqMode = arg(10, 0);
    const int64_t kvqMode = arg(11, 0);

    MlaPrologV3Tensors t;
    t.tokenX = {T, He};
    t.weightDq = {He, Hcq};
    t.weightUqQr = {Hcq, N * (D + Dr)};
    t.weightUk = {N, D, Hckv};
    t.weightDkvKr = {He, Hckv + Dr};
    t.gammaCq = {Hcq};
    t.gammaCkv = {Hckv};
    t.ropeSin = {T, Dr};
    t.ropeCos = {T, Dr};
    t.kvCache = {blockNum, blockSize, 1, Hckv};
    t.krCache = {blockNum, blockSize, 1, Dr};
    t.cacheIndex = {T};

    t.query = {T, N, Hckv};
    t.queryRope = {T, N, Dr};
    t.kvCacheOut = t.kvCache;
    t.krCacheOut = t.krCache;
    t.dequantScaleQNope = {0};
    t.queryNorm = {T, Hcq};
    t.dequantScaleQNorm = {0};

    t.weightQuantMode = wqMode;
    t.kvQuantMode = kvqMode;
    t.queryNormFlag = true;
    t.cacheMode = "PA_BSND";

    if (wqMode != 0) {
        t.dequantScaleWUqQr = {1, N * (D + Dr)};
    }
    if (wqMode == 2 || wqMode == 3 || wqMode == 4 || wqMode == 5) {
        t.dequantScaleX = {T, 1};
        t.dequantScaleWDq = {1, Hcq};
        t.dequantScaleWDkvKr = {1, Hckv + Dr};
        t.tokenXType = (wqMode == 3) ? ge::DT_FLOAT8_E4M3FN : ge::DT_INT8;
        t.weightDqType = (wqMode == 3) ? ge::DT_FLOAT8_E4M3FN : ge::DT_INT8;
        t.weightUqQrType = (wqMode == 3) ? ge::DT_FLOAT8_E4M3FN : ge::DT_INT8;
        t.weightDkvKrType = (wqMode == 3) ? ge::DT_FLOAT8_E4M3FN : ge::DT_INT8;
        if (wqMode == 3) {
            // MXFP8: weight scales are E8M0, laid out as [out, in/32]
            t.scaleXType = ge::DT_FLOAT8_E8M0;
            t.scaleWDqType = ge::DT_FLOAT8_E8M0;
            t.scaleWUqQrType = ge::DT_FLOAT8_E8M0;
            t.scaleWDkvKrType = ge::DT_FLOAT8_E8M0;
            t.dequantScaleX = {T, He / 32};
            t.dequantScaleWDq = {Hcq, He / 32};
            t.dequantScaleWUqQr = {N * (D + Dr), Hcq / 32};
            t.dequantScaleWDkvKr = {Hckv + Dr, He / 32};
            t.queryNormType = ge::DT_FLOAT8_E4M3FN;
            t.queryNorm = {T, Hcq};
            t.dequantScaleQNorm = {T, Hcq / 32};
            t.scaleQNormType = ge::DT_FLOAT8_E8M0;
            t.scaleQNopeType = ge::DT_FLOAT;
        }
    }

    optiling::MlaPrologTilingData tilingData{};
    MlaPrologV3TilingResult res;
    std::string err;
    const bool ok = MlaPrologV3RunTiling(t, &tilingData, &res, &err);
    if (!ok) {
        std::printf("TILING FAILED: %s\n", err.c_str());
        return 1;
    }
    std::printf("tilingKey=%llu blockDim=%u workspace=%zu\n", (unsigned long long)res.tilingKey, res.blockDim,
                res.workspaceSize);
    std::printf("tplArgs: cacheMode=%u scenario=%u quantMode=%u dequantOpt=%u groupOpt=%u empty=%u seqLen=%u "
                "splitM=%u rope=%u cv=%u\n",
                res.cacheMode, res.scenario, res.quantMode, res.enableDequantOpt, res.enableGroupComputeOpt,
                res.emptyTensorMode, res.actualSeqLenMode, res.splitMMode, res.enableRope, res.cvMode);
    std::printf("baseParams: tokenSize=%u stepBatch=%u numHead=%u dimQc=%u dimRope=%u hcq=%u hckv=%u\n",
                tilingData.baseParams.tokenSize, tilingData.baseParams.stepBatchSize,
                tilingData.baseParams.numHeadSize, tilingData.baseParams.dimHeadSizeQc,
                tilingData.baseParams.dimHeadRope, tilingData.baseParams.headSizeCq,
                tilingData.baseParams.headSizeCkv);
    std::printf("blocks: mm1=%u mm2=%u mm3=%u mm4=%u vector=%u\n", tilingData.baseParams.mm1BlockNum,
                tilingData.baseParams.mm2BlockNum, tilingData.baseParams.mm3BlockNum, tilingData.baseParams.mm4BlockNum,
                tilingData.baseParams.vectorBlockNum);
    return 0;
}
