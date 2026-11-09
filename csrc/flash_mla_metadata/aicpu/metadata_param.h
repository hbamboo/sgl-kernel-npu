// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
//
// Launch ABI for the package-owned FlashMLA metadata AICPU kernel.
// The first 20 bytes match CANN's aicpu::AicpuParamHead (packed) so the ACL
// runtime can copy the host buffer to device HBM. The kernel entry
// RunCpuKernel(void*) is registered directly with aclrtRegisterCpuFunc; it
// copies this header and reads all device pointers from the device copy.
// Scalar attributes are launch-time constants and are captured into the graph
// task. Sequence-length values are always read from device tensors by the
// kernel, including on every graph replay.

#ifndef SGL_FLASH_MLA_METADATA_PARAM_H
#define SGL_FLASH_MLA_METADATA_PARAM_H

#include <cstddef>
#include <cstdint>

constexpr uint32_t kSglFlashMlaMetadataMagic = 0x4D4C4653U; // "SFLM"
constexpr uint32_t kSglFlashMlaMetadataVersion = 1U;
constexpr uint32_t kSglFlashMlaMetadataIoCount = 4U;

#pragma pack(push, 1)
struct SglFlashMlaMetadataParam {
    // CANN aicpu::AicpuParamHead compatible prefix.
    uint32_t length;
    uint32_t ioAddrNum;
    uint32_t extInfoLength;
    uint64_t extInfoAddr;

    // Device addresses immediately after the standard 20-byte head, matching
    // the CANN CPU-kernel argument protocol's io address array.
    // cuSeqlensQ and sequsedQ are 0 when not provided.
    uint64_t cuSeqlensQ;
    uint64_t cacheSeqlens;
    uint64_t sequsedQ;
    uint64_t metadata;

    uint32_t magic;
    uint32_t version;

    int64_t metadataWords; // capacity of metadata in uint32 words
    int32_t batchSize;     // elements in cacheSeqlens
    int32_t maxSeqlenQ;
    int32_t maxSeqlenKv;
    int32_t numHeadsQ;
    int32_t numHeadsKv;
    int32_t headDimQk;
    int32_t headDimV;
    int32_t maskMode;
    int32_t aicCoreNum;
    int32_t aivCoreNum;
    int64_t winLeft;
    int64_t winRight;
    char layoutQ[8]; // NUL-terminated: BSND, BNSD or TND
};
#pragma pack(pop)

static_assert(offsetof(SglFlashMlaMetadataParam, cuSeqlensQ) == 20,
              "io addresses must immediately follow the 20-byte AicpuParamHead");
static_assert(offsetof(SglFlashMlaMetadataParam, cacheSeqlens) == 28,
              "cacheSeqlens must be the second io address");
static_assert(offsetof(SglFlashMlaMetadataParam, sequsedQ) == 36,
              "sequsedQ must be the third io address");
static_assert(offsetof(SglFlashMlaMetadataParam, metadata) == 44,
              "metadata must be the fourth io address");
static_assert(offsetof(SglFlashMlaMetadataParam, magic) == 52,
              "magic follows the io address array");
static_assert(sizeof(SglFlashMlaMetadataParam) == 132,
              "launch args ABI size is fixed for host/device agreement");
static_assert(sizeof(SglFlashMlaMetadataParam) <= 4096,
              "metadata launch args must fit the runtime host-args budget");

#endif // SGL_FLASH_MLA_METADATA_PARAM_H
