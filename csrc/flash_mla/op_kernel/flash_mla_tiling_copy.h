#pragma once
#include "arch35/flash_mla_tiling_data.h"
namespace optiling {
__aicore__ inline void CopyFlashMlaTiling(GM_ADDR ptr, FlashMlaLaunchData& dst) {
    const __gm__ FlashMlaLaunchData* src = reinterpret_cast<const __gm__ FlashMlaLaunchData*>(ptr);
    dst.baseTiling.flashMlaBaseParams.bSize = src->baseTiling.flashMlaBaseParams.bSize;
    dst.baseTiling.flashMlaBaseParams.t1Size = src->baseTiling.flashMlaBaseParams.t1Size;
    dst.baseTiling.flashMlaBaseParams.t2Size = src->baseTiling.flashMlaBaseParams.t2Size;
    dst.baseTiling.flashMlaBaseParams.n2Size = src->baseTiling.flashMlaBaseParams.n2Size;
    dst.baseTiling.flashMlaBaseParams.gSize = src->baseTiling.flashMlaBaseParams.gSize;
    dst.baseTiling.flashMlaBaseParams.s1Size = src->baseTiling.flashMlaBaseParams.s1Size;
    dst.baseTiling.flashMlaBaseParams.s2Size = src->baseTiling.flashMlaBaseParams.s2Size;
    dst.baseTiling.flashMlaBaseParams.dSize = src->baseTiling.flashMlaBaseParams.dSize;
    dst.baseTiling.flashMlaBaseParams.dSizeV = src->baseTiling.flashMlaBaseParams.dSizeV;
    dst.baseTiling.flashMlaBaseParams.dSizeRope = src->baseTiling.flashMlaBaseParams.dSizeRope;
    dst.baseTiling.flashMlaBaseParams.actualSeqLengthsQSize = src->baseTiling.flashMlaBaseParams.actualSeqLengthsQSize;
    dst.baseTiling.flashMlaBaseParams.actualSeqLengthsKVSize = src->baseTiling.flashMlaBaseParams.actualSeqLengthsKVSize;
    dst.baseTiling.flashMlaBaseParams.scaleValue = src->baseTiling.flashMlaBaseParams.scaleValue;
    dst.baseTiling.flashMlaBaseParams.isKvContinuous = src->baseTiling.flashMlaBaseParams.isKvContinuous;
    dst.baseTiling.flashMlaBaseParams.isSoftMaxLseEnable = src->baseTiling.flashMlaBaseParams.isSoftMaxLseEnable;
    dst.baseTiling.flashMlaBaseParams.l2CacheOffFlag = src->baseTiling.flashMlaBaseParams.l2CacheOffFlag;
    dst.baseTiling.flashMlaBaseParams.coreNum = src->baseTiling.flashMlaBaseParams.coreNum;
    dst.baseTiling.flashMlaBaseParams.outputLayout = src->baseTiling.flashMlaBaseParams.outputLayout;
    dst.baseTiling.flashMlaAttenMaskParams.sparseMode = src->baseTiling.flashMlaAttenMaskParams.sparseMode;
    dst.baseTiling.flashMlaAttenMaskParams.preTokens = src->baseTiling.flashMlaAttenMaskParams.preTokens;
    dst.baseTiling.flashMlaAttenMaskParams.nextTokens = src->baseTiling.flashMlaAttenMaskParams.nextTokens;
    dst.baseTiling.flashMlaAttenMaskParams.attenMaskBatch = src->baseTiling.flashMlaAttenMaskParams.attenMaskBatch;
    dst.baseTiling.flashMlaAttenMaskParams.attenMaskS1Size = src->baseTiling.flashMlaAttenMaskParams.attenMaskS1Size;
    dst.baseTiling.flashMlaAttenMaskParams.attenMaskS2Size = src->baseTiling.flashMlaAttenMaskParams.attenMaskS2Size;
    dst.baseTiling.flashMlaAttenMaskParams.isRowInvalidOpen = src->baseTiling.flashMlaAttenMaskParams.isRowInvalidOpen;
    dst.baseTiling.flashMlaAttenMaskParams.isExistRowInvalid = src->baseTiling.flashMlaAttenMaskParams.isExistRowInvalid;
    dst.baseTiling.flashMlaPseParams.pseShiftByBatch = src->baseTiling.flashMlaPseParams.pseShiftByBatch;
    dst.baseTiling.flashMlaPseParams.pseS1Size = src->baseTiling.flashMlaPseParams.pseS1Size;
    dst.baseTiling.flashMlaPseParams.pseS2Size = src->baseTiling.flashMlaPseParams.pseS2Size;
    dst.baseTiling.flashMlaPseParams.pseStride = src->baseTiling.flashMlaPseParams.pseStride;
    dst.baseTiling.flashMlaPseParams.qStartIdx = src->baseTiling.flashMlaPseParams.qStartIdx;
    dst.baseTiling.flashMlaPseParams.kvStartIdx = src->baseTiling.flashMlaPseParams.kvStartIdx;
    dst.baseTiling.flashMlaSystemPrefixParams.isActualSharedPrefixLenNull = src->baseTiling.flashMlaSystemPrefixParams.isActualSharedPrefixLenNull;
    dst.baseTiling.flashMlaSystemPrefixParams.prefixSeqInnerSize = src->baseTiling.flashMlaSystemPrefixParams.prefixSeqInnerSize;
    dst.baseTiling.flashMlaPageAttentionParams.paLayoutType = src->baseTiling.flashMlaPageAttentionParams.paLayoutType;
    dst.baseTiling.flashMlaPageAttentionParams.blockSize = src->baseTiling.flashMlaPageAttentionParams.blockSize;
    dst.baseTiling.flashMlaPageAttentionParams.maxBlockNumPerBatch = src->baseTiling.flashMlaPageAttentionParams.maxBlockNumPerBatch;
    dst.baseTiling.flashMlaLeftPaddingParams.isQHasLeftPadding = src->baseTiling.flashMlaLeftPaddingParams.isQHasLeftPadding;
    dst.baseTiling.flashMlaLeftPaddingParams.isKVHasLeftPadding = src->baseTiling.flashMlaLeftPaddingParams.isKVHasLeftPadding;
    dst.baseTiling.flashMlaPostQuantParams.isPostQuantPerChnl = src->baseTiling.flashMlaPostQuantParams.isPostQuantPerChnl;
    dst.baseTiling.flashMlaPostQuantParams.isPostQuantBF16 = src->baseTiling.flashMlaPostQuantParams.isPostQuantBF16;
    dst.baseTiling.flashMlaWorkspaceParams.accumOutSize = src->baseTiling.flashMlaWorkspaceParams.accumOutSize;
    dst.baseTiling.flashMlaWorkspaceParams.logSumExpSize = src->baseTiling.flashMlaWorkspaceParams.logSumExpSize;
    dst.baseTiling.flashMlaEmptyTensorParams.singleCoreSize = src->baseTiling.flashMlaEmptyTensorParams.singleCoreSize;
    dst.baseTiling.flashMlaEmptyTensorParams.needInit = src->baseTiling.flashMlaEmptyTensorParams.needInit;
    dst.baseTiling.flashMlaEmptyTensorParams.totalOutputSize = src->baseTiling.flashMlaEmptyTensorParams.totalOutputSize;
    dst.baseTiling.flashMlaEmptyTensorParams.totalSoftMaxLseOutputSize = src->baseTiling.flashMlaEmptyTensorParams.totalSoftMaxLseOutputSize;
    dst.baseTiling.flashMlaBaseParams.keyStrides.bnStride = src->baseTiling.flashMlaBaseParams.keyStrides.bnStride;
    dst.baseTiling.flashMlaBaseParams.keyStrides.n2Stride = src->baseTiling.flashMlaBaseParams.keyStrides.n2Stride;
    dst.baseTiling.flashMlaBaseParams.valueStrides.bnStride = src->baseTiling.flashMlaBaseParams.valueStrides.bnStride;
    dst.baseTiling.flashMlaBaseParams.valueStrides.n2Stride = src->baseTiling.flashMlaBaseParams.valueStrides.n2Stride;
    dst.baseTiling.flashMlaBaseParams.kRopeStrides.bnStride = src->baseTiling.flashMlaBaseParams.kRopeStrides.bnStride;
    dst.baseTiling.flashMlaBaseParams.kRopeStrides.n2Stride = src->baseTiling.flashMlaBaseParams.kRopeStrides.n2Stride;
    dst.baseTiling.flashMlaBaseParams.kScaleStrides.bnStride = src->baseTiling.flashMlaBaseParams.kScaleStrides.bnStride;
    dst.baseTiling.flashMlaBaseParams.kScaleStrides.n2Stride = src->baseTiling.flashMlaBaseParams.kScaleStrides.n2Stride;
    dst.baseTiling.flashMlaBaseParams.vScaleStrides.bnStride = src->baseTiling.flashMlaBaseParams.vScaleStrides.bnStride;
    dst.baseTiling.flashMlaBaseParams.vScaleStrides.n2Stride = src->baseTiling.flashMlaBaseParams.vScaleStrides.n2Stride;
    dst.dtype=src->dtype;
    dst.layout=src->layout;
    dst.kvLayout=src->kvLayout;
    dst.hasMask=src->hasMask;
}
}
