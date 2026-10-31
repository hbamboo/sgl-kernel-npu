// Field order/types retained from upstream ScatterPaKvCacheTilingData.
#pragma once
#include <cstdint>
struct ScatterPaKvCacheTilingData {
    int64_t usedCoreNum;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t kHandleNumPerCore;
    int64_t vHandleNumPerCore;
    int64_t kTailHandleNum;
    int64_t vTailHandleNum;
    int64_t kLoopNum;
    int64_t vLoopNum;
    int64_t kHandleNumPerLoop;
    int64_t vHandleNumPerLoop;
    int64_t keyStride0;
    int64_t keyStride1;
    int64_t keyStride2;
    int64_t valueStride0;
    int64_t valueStride1;
    int64_t valueStride2;
    int64_t kHeadSize;
    int64_t vHeadSize;
    int64_t batch;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t seqLen;
    int64_t numHead;
    int64_t numTokens;
    int64_t ubSize;
    int64_t kStride;
    int64_t vStride;
    int64_t kOffset;
    int64_t vOffset;
    int64_t keyCacheStride0;
    int64_t keyCacheStride1;
    int64_t keyCacheStride2;
    int64_t keyCacheStride3;
    int64_t valueCacheStride0;
    int64_t valueCacheStride1;
    int64_t valueCacheStride2;
    int64_t valueCacheStride3;
    int64_t kCacheBlockStride;
    int64_t vCacheBlockStride;
};
struct ScatterPaKvCacheLaunchData { ScatterPaKvCacheTilingData data; uint64_t key; uint64_t dtype; uint64_t indexBytes; };
