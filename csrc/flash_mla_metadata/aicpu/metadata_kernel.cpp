// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
//
// Package-owned AICPU implementation of FlashMLA metadata generation.
// The scheduling algorithm is the original FlashMlaMetadataCpuKernel +
// load_balance::SectionStreamK (96x112 tiling, section/FA/FD layout), migrated
// into sgl-kernel-npu. The only adaptation is the launch ABI: instead of the
// CANN CpuKernelContext/NodeDef path, this library exports RunCpuKernel(void*)
// and reads an SglFlashMlaMetadataParam built by flash_mla_metadata.cpp.
// Sequence lengths are read from device HBM on every launch and every graph
// replay; no host D2H copy or captured metadata content is used.

#include "metadata_param.h"
#include "flash_mla_metadata.h"
#include "load_balance/section_stream_k/section_stream_k.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace optiling;
using load_balance::BaseInfo;
using load_balance::DataType;
using load_balance::DeviceInfo;
using load_balance::Layout;
using load_balance::SectionStreamK;
using load_balance::SectionStreamKParam;
using load_balance::SectionStreamKResult;
using load_balance::SparseMode;

namespace {

constexpr uint32_t kStatusOk = 0U;
constexpr uint32_t kStatusParamInvalid = 1U;
constexpr uint32_t kStatusCapacity = 2U;
constexpr uint32_t kStatusSchedule = 3U;

constexpr uint32_t kFaKernelStatusOk = 0U;
constexpr uint32_t kFaKernelStatusParamInvalid = 1U;

void LogError(const char *message, int64_t value)
{
    std::fprintf(stderr, "[sgl_flash_mla_metadata] %s: %lld\n", message, static_cast<long long>(value));
}

std::vector<int64_t> ReadDeviceInt32(uint64_t address, int64_t count)
{
    std::vector<int64_t> result{};
    if (address == 0U || count <= 0) {
        return result;
    }
    const int32_t *source = reinterpret_cast<const int32_t *>(static_cast<uintptr_t>(address));
    result.resize(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        result[static_cast<size_t>(i)] = static_cast<int64_t>(source[i]);
    }
    return result;
}

class FlashMlaMetadataAicpuKernel {
public:
    uint32_t Run(const SglFlashMlaMetadataParam &args)
    {
        if (!Prepare(args)) {
            return kStatusParamInvalid;
        }

        SectionStreamKResult splitRes{};
        if (!BalanceSchedule(splitRes)) {
            return kStatusSchedule;
        }

        if (!CheckMetadataCapacity(splitRes.sectionNum)) {
            return kStatusCapacity;
        }

        if (!GenMetadata(splitRes)) {
            return kStatusParamInvalid;
        }
        return kStatusOk;
    }

private:
    bool Prepare(const SglFlashMlaMetadataParam &args)
    {
        args_ = &args;
        metadata_ = reinterpret_cast<FA_METADATA_T *>(static_cast<uintptr_t>(args.metadata));
        if (metadata_ == nullptr || args.metadataWords <= 0) {
            LogError("metadata output is empty", 0);
            return false;
        }
        if (!CheckAttrs()) {
            return false;
        }
        if (!CheckActualKvSeq()) {
            return false;
        }
        if (actualSeqlenKv_.empty()) {
            LogError("cacheSeqlens must be provided to derive batch_size", 0);
            return false;
        }
        if (!CheckActualQuerySeq(static_cast<int64_t>(actualSeqlenKv_.size()))) {
            return false;
        }
        InitDeviceInfo();
        InitBaseInfo();
        InitLoadBalanceParams();
        return true;
    }

    bool CheckAttrs()
    {
        if (!(args_->numHeadsQ >= 1 && args_->numHeadsQ <= 128)) {
            LogError("numHeadsQ must be in [1,128]", args_->numHeadsQ);
            return false;
        }
        if (args_->numHeadsKv != FlashMlaCommon::MLA_KV_HEADS) {
            LogError("MLA kv head num must be 1", args_->numHeadsKv);
            return false;
        }
        if (args_->headDimQk != FlashMlaCommon::MLA_HEAD_DIM_QK) {
            LogError("MLA headDimQk must be 576", args_->headDimQk);
            return false;
        }
        if (args_->headDimV != FlashMlaCommon::MLA_HEAD_DIM_V) {
            LogError("MLA headDimV must be 512", args_->headDimV);
            return false;
        }
        if (!(args_->maskMode == 0 || args_->maskMode == 3 || args_->maskMode == 4)) {
            LogError("maskMode only supports 0, 3, 4", args_->maskMode);
            return false;
        }
        if (!(args_->winLeft >= -1 && args_->winLeft <= std::numeric_limits<int32_t>::max() &&
              args_->winRight >= -1 && args_->winRight <= std::numeric_limits<int32_t>::max())) {
            LogError("win_left/win_right out of range", args_->winLeft);
            return false;
        }
        if (!(args_->maskMode == 4 || (args_->winLeft == -1 && args_->winRight == -1))) {
            LogError("win_left/win_right must be -1 for mask_mode 0 or 3", 0);
            return false;
        }
        layoutQ_ = std::string(args_->layoutQ, strnlen(args_->layoutQ, sizeof(args_->layoutQ)));
        if (!(layoutQ_ == "TND" || layoutQ_ == "BNSD" || layoutQ_ == "BSND")) {
            LogError("layoutQ only supports TND, BNSD, BSND", 0);
            return false;
        }
        if (!(args_->aicCoreNum > 0)) {
            LogError("aicCoreNum must be positive", args_->aicCoreNum);
            return false;
        }
        if (!(args_->aivCoreNum > 0)) {
            LogError("aivCoreNum must be positive", args_->aivCoreNum);
            return false;
        }
        return true;
    }

    bool CheckMetadataCapacity(uint32_t sectionNum)
    {
        const uint64_t maxSections =
            static_cast<uint64_t>(actualSeqlenKv_.size()) * static_cast<uint64_t>(args_->numHeadsKv);
        if (!(sectionNum > 0U && sectionNum <= maxSections)) {
            LogError("sectionNum out of range", sectionNum);
            return false;
        }
        const int64_t needWords =
            static_cast<int64_t>(FlashMlaCommon::HEAD_METADATA_STRIDE) +
            static_cast<int64_t>(sectionNum) *
                (static_cast<int64_t>(args_->aicCoreNum) * FlashMlaCommon::FA_METADATA_STRIDE +
                 static_cast<int64_t>(args_->aivCoreNum) * FlashMlaCommon::FD_METADATA_STRIDE);
        if (args_->metadataWords < needWords) {
            LogError("metadata buffer too small, need words", needWords);
            return false;
        }
        return true;
    }

    bool CheckActualQuerySeq(int64_t batchSize)
    {
        isActualSeqlenQAccum_ = false;
        actualSeqlenQ_.clear();
        std::vector<int64_t> cuSeqlensQ = ReadDeviceInt32(args_->cuSeqlensQ, args_->cuSeqlensQ == 0U ? 0 : batchSize + 1);
        std::vector<int64_t> sequsedQ = ReadDeviceInt32(args_->sequsedQ, args_->sequsedQ == 0U ? 0 : batchSize);

        if (!cuSeqlensQ.empty() && static_cast<int64_t>(cuSeqlensQ.size()) != batchSize + 1) {
            LogError("cuSeqlensQ shape must be batchSize+1", static_cast<int64_t>(cuSeqlensQ.size()));
            return false;
        }
        if (!sequsedQ.empty() && static_cast<int64_t>(sequsedQ.size()) != batchSize) {
            LogError("sequsedQ shape must be batchSize", static_cast<int64_t>(sequsedQ.size()));
            return false;
        }

        if (layoutQ_ == "TND") {
            if (cuSeqlensQ.empty()) {
                LogError("TND requires cuSeqlensQ", 0);
                return false;
            }
        } else if (!cuSeqlensQ.empty()) {
            LogError("non-TND layout must not provide cuSeqlensQ", 0);
            return false;
        }

        for (size_t i = 0; i < sequsedQ.size(); ++i) {
            if (sequsedQ[i] < 0) {
                LogError("sequsedQ element must be non-negative", sequsedQ[i]);
                return false;
            }
        }
        if (!cuSeqlensQ.empty() && cuSeqlensQ[0] != 0) {
            LogError("cuSeqlensQ[0] must be 0", cuSeqlensQ[0]);
            return false;
        }
        for (size_t i = 1; i < cuSeqlensQ.size(); ++i) {
            if (cuSeqlensQ[i] < cuSeqlensQ[i - 1]) {
                LogError("cuSeqlensQ must be non-decreasing", cuSeqlensQ[i]);
                return false;
            }
            if (!sequsedQ.empty() && sequsedQ[i - 1] > cuSeqlensQ[i] - cuSeqlensQ[i - 1]) {
                LogError("sequsedQ exceeds physical query length", sequsedQ[i - 1]);
                return false;
            }
        }

        if (!sequsedQ.empty()) {
            isActualSeqlenQAccum_ = false;
            actualSeqlenQ_ = sequsedQ;
        } else if (!cuSeqlensQ.empty()) {
            isActualSeqlenQAccum_ = true;
            actualSeqlenQ_.assign(cuSeqlensQ.begin() + 1, cuSeqlensQ.end());
        }
        return true;
    }

    bool CheckActualKvSeq()
    {
        isActualSeqlenKvAccum_ = false;
        actualSeqlenKv_.clear();
        std::vector<int64_t> cacheSeqlens = ReadDeviceInt32(args_->cacheSeqlens, args_->batchSize);
        for (size_t i = 0; i < cacheSeqlens.size(); ++i) {
            if (cacheSeqlens[i] < 0) {
                LogError("cacheSeqlens element must be non-negative", cacheSeqlens[i]);
                return false;
            }
        }
        actualSeqlenKv_ = cacheSeqlens;
        return true;
    }

    void InitDeviceInfo()
    {
        deviceInfo_.aicCoreMaxNum = static_cast<uint32_t>(args_->aicCoreNum);
        deviceInfo_.aivCoreMaxNum = static_cast<uint32_t>(args_->aivCoreNum);
        deviceInfo_.aicCoreMinNum = static_cast<uint32_t>(args_->aicCoreNum);
        deviceInfo_.aivCoreMinNum = static_cast<uint32_t>(args_->aivCoreNum);
    }

    void InitLoadBalanceParams()
    {
        param_.mBaseSize = 96;
        param_.s2BaseSize = 112;
        // M is below two base blocks: multiple sections only add sync overhead.
        {
            const uint32_t g = static_cast<uint32_t>(args_->numHeadsQ) / static_cast<uint32_t>(args_->numHeadsKv);
            const uint32_t m = g * static_cast<uint32_t>(args_->maxSeqlenQ);
            if (m <= param_.mBaseSize * 2U) {
                param_.l2Byte = 0U; // disable section splitting
            } else {
                param_.l2Byte = 96U * 1024U * 1024U; // 96MB
            }
        }
        param_.fdTolerance = 10;
        param_.fdLeastBlock = 3;
        param_.fdOn = true;
    }

    void InitBaseInfo()
    {
        baseInfo_.batchSize = static_cast<uint32_t>(actualSeqlenKv_.size());
        baseInfo_.querySeqSize = static_cast<uint32_t>(args_->maxSeqlenQ);
        baseInfo_.queryHeadNum = static_cast<uint32_t>(args_->numHeadsQ);
        baseInfo_.kvSeqSize = static_cast<uint32_t>(args_->maxSeqlenKv);
        baseInfo_.kvHeadNum = static_cast<uint32_t>(args_->numHeadsKv);
        baseInfo_.headDimQk = static_cast<uint32_t>(args_->headDimQk);
        baseInfo_.headDimV = static_cast<uint32_t>(args_->headDimV);
        SparseMode maskMode = SparseMode::BUTT;
        if (args_->maskMode != 0) {
            maskMode = static_cast<SparseMode>(args_->maskMode);
        }
        baseInfo_.attenMaskFlag = (maskMode != SparseMode::BUTT);
        baseInfo_.sparseMode = static_cast<uint32_t>(maskMode);
        baseInfo_.preToken =
            args_->winLeft == -1 ? std::numeric_limits<uint32_t>::max() : static_cast<int64_t>(args_->winLeft);
        baseInfo_.nextToken =
            args_->winRight == -1 ? std::numeric_limits<uint32_t>::max() : static_cast<int64_t>(args_->winRight);
        baseInfo_.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
        baseInfo_.queryType = DataType::FP16;
        baseInfo_.kvType = DataType::FP16;
        baseInfo_.isCumulativeKvSeq = isActualSeqlenKvAccum_;
        baseInfo_.actualKvSeqSize = actualSeqlenKv_;
        baseInfo_.isCumulativeQuerySeq = isActualSeqlenQAccum_;
        baseInfo_.actualQuerySeqSize = actualSeqlenQ_;
    }

    bool BalanceSchedule(SectionStreamKResult &splitRes)
    {
        return SectionStreamK::Compute(deviceInfo_, baseInfo_, param_, splitRes) == SECTION_STREAM_K_SUCCESS;
    }

    bool GenMetadata(const SectionStreamKResult &splitRes)
    {
        detail::FaMetadata faMetadata(metadata_, splitRes.sectionNum, static_cast<uint32_t>(args_->aicCoreNum),
                                      static_cast<uint32_t>(args_->aivCoreNum));
        faMetadata.Clear();

        faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
        faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, HEAD_M_BASE_SIZE_MLA);
        faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, HEAD_S2_BASE_SIZE_MLA);
        faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, static_cast<FA_METADATA_T>(args_->aicCoreNum));
        faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, static_cast<FA_METADATA_T>(args_->aivCoreNum));
        faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX,
                                   static_cast<FA_METADATA_T>(load_balance::KernelSplitMode::BN2_S1G_S2));
        if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                        [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
            faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
        }

        load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(args_->aicCoreNum)};
        for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
            const auto &faRes = splitRes.sectionFaResult[secIdx];
            for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
                const auto &prevFaRes =
                    (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
                const uint32_t prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
                FA_METADATA_T bn2Start =
                    (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
                FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
                FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bNEnd[aicIdx]);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
                faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                         faRes.firstFdDataWorkspaceIdx[aicIdx]);
            }

            const auto &fdRes = splitRes.sectionFdResult[secIdx];
            for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
                const uint32_t task = fdRes.taskIdx[aivIdx];
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bNIdx[task]);
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[task]);
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[task]);
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[task]);
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
                faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
            }
        }
        return true;
    }

    const SglFlashMlaMetadataParam *args_{nullptr};
    FA_METADATA_T *metadata_{nullptr};

    std::string layoutQ_{"BSND"};
    bool isActualSeqlenQAccum_{false};
    bool isActualSeqlenKvAccum_{false};
    std::vector<int64_t> actualSeqlenQ_{};
    std::vector<int64_t> actualSeqlenKv_{};
    DeviceInfo deviceInfo_{};
    BaseInfo baseInfo_{};
    SectionStreamKParam param_{};
};

} // namespace

extern "C" __attribute__((visibility("default"))) uint32_t RunCpuKernel(void *param)
{
    if (param == nullptr) {
        return kFaKernelStatusParamInvalid;
    }
    SglFlashMlaMetadataParam args;
    std::memcpy(&args, param, sizeof(args));
    if (args.magic != kSglFlashMlaMetadataMagic || args.version != kSglFlashMlaMetadataVersion ||
        args.length < sizeof(SglFlashMlaMetadataParam) || args.ioAddrNum != kSglFlashMlaMetadataIoCount) {
        LogError("invalid launch args header", static_cast<int64_t>(args.magic));
        return kFaKernelStatusParamInvalid;
    }

    FlashMlaMetadataAicpuKernel kernel;
    return kernel.Run(args);
}
