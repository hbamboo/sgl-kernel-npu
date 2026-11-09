// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include <acl/acl.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <link.h>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "flash_mla_metadata_api.h"
#include "aicpu/metadata_param.h"

namespace sglang::npu_kernel {
using Optional=c10::optional<at::Tensor>;
namespace {
// The metadata producer is a package-owned AICPU shared object shipped next to
// this host library in the wheel. It is loaded through ACL CPU-kernel mode and
// launched on the current NPU stream; no vendor OPP registration is involved.
struct MetadataKernel {
    aclrtBinHandle binary{nullptr};
    aclrtFuncHandle function{nullptr};
    std::vector<char> image{};
};

std::mutex g_metadata_kernels_mutex;
std::map<int, MetadataKernel> g_metadata_kernels;

struct ImageMatch {
    uintptr_t address{0};
    std::string name{};
    bool found{false};
};

int MatchImage(struct dl_phdr_info* info, size_t, void* data) {
    auto* match = static_cast<ImageMatch*>(data);
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const auto& header = info->dlpi_phdr[i];
        if (header.p_type != PT_LOAD) {
            continue;
        }
        const uintptr_t start = static_cast<uintptr_t>(info->dlpi_addr) + static_cast<uintptr_t>(header.p_vaddr);
        const uintptr_t end = start + static_cast<uintptr_t>(header.p_memsz);
        if (match->address >= start && match->address < end) {
            if (info->dlpi_name != nullptr) {
                match->name = info->dlpi_name;
            }
            match->found = true;
            return 1;
        }
    }
    return 0;
}

std::string MetadataLibraryPath() {
    // Resolve the directory of libsgl_kernel_npu_flash_mla.so without any build
    // path or environment dependency. dl_iterate_phdr is symbol-visibility
    // independent (the release build uses -fvisibility=hidden).
    ImageMatch match;
    match.address = reinterpret_cast<uintptr_t>(&flash_mla_metadata_impl);
    dl_iterate_phdr(MatchImage, &match);
    std::string hostImage = match.found ? match.name : std::string();
    std::string directory = ".";
    const auto slash = hostImage.find_last_of('/');
    if (slash != std::string::npos) {
        directory = hostImage.substr(0, slash);
    }
    return directory + "/libsgl_flash_mla_metadata_aicpu.so";
}

MetadataKernel& GetMetadataKernel(int device) {
    std::lock_guard<std::mutex> lock(g_metadata_kernels_mutex);
    auto it = g_metadata_kernels.find(device);
    if (it != g_metadata_kernels.end()) {
        return it->second;
    }

    const std::string path = MetadataLibraryPath();
    std::ifstream file(path, std::ios::binary);
    TORCH_CHECK(file, "FlashMLA metadata AICPU library not found: ", path,
                ". Rebuild/install the sgl_kernel_npu wheel with its companion library.");

    MetadataKernel kernel;
    kernel.image.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    TORCH_CHECK(!kernel.image.empty(), "FlashMLA metadata AICPU library is empty: ", path);

    aclrtBinaryLoadOption option;
    std::memset(&option, 0, sizeof(option));
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 2; // load CPU kernel shared object from data
    aclrtBinaryLoadOptions options = {&option, 1};
    const aclError loadRc =
        aclrtBinaryLoadFromData(kernel.image.data(), kernel.image.size(), &options, &kernel.binary);
    TORCH_CHECK(loadRc == ACL_SUCCESS, "Cannot load FlashMLA metadata AICPU library ", path, ": ", loadRc);
    const aclError registerRc =
        aclrtRegisterCpuFunc(kernel.binary, "RunCpuKernel", "SglFlashMlaMetadata", &kernel.function);
    TORCH_CHECK(registerRc == ACL_SUCCESS, "Cannot register FlashMLA metadata AICPU function: ", registerRc);

    return g_metadata_kernels.emplace(device, std::move(kernel)).first->second;
}

} // namespace

at::Tensor flash_mla_metadata_impl(const at::Tensor& seq,int64_t heads,int64_t kvheads,
    const Optional& cu,const Optional& used,const c10::optional<int64_t>& maxq_,
    const c10::optional<int64_t>& maxkv_,int64_t dqk,int64_t dv,
    const c10::optional<int64_t>& mode_,int64_t left,int64_t right,
    const c10::optional<std::string>& layout_) {
    const c10::DeviceGuard guard(seq.device());
    auto maxq=maxq_.value_or(-1),maxkv=maxkv_.value_or(-1),mode=mode_.value_or(0);
    auto layout=layout_.value_or("BSND");

    // Pure Host contract from the original ACLNN FlashMlaMetadataCheck. All
    // attribute/meta checks complete before the platform query, output
    // allocation, or AICPU enqueue, so invalid inputs fail synchronously and
    // cannot poison the device context. Tensor VALUE checks stay on AICPU.
    TORCH_CHECK(seq.device().type()==c10::DeviceType::PrivateUse1 && seq.scalar_type()==at::kInt &&
                    seq.dim()==1 && seq.is_contiguous(),
                "cache_seqlens must be a contiguous NPU int32 vector");
    TORCH_CHECK(layout=="BSND"||layout=="BNSD"||layout=="TND",
                "layout_q only supports TND, BNSD, BSND, but got ",layout);
    TORCH_CHECK(maxq==-1 || (maxq>=0 && maxq<=std::numeric_limits<int32_t>::max()),
                "max_seqlen_q must be -1 or non-negative and fit int32");
    TORCH_CHECK(maxkv==-1 || (maxkv>=0 && maxkv<=std::numeric_limits<int32_t>::max()),
                "max_seqlen_kv must be -1 or non-negative and fit int32");
    TORCH_CHECK(heads>=1 && heads<=128,"num_heads_q must be in [1, 128]");
    TORCH_CHECK(kvheads==1,"MLA kv head num must be 1");
    TORCH_CHECK(dqk==576 && dv==512,"MLA requires head_dim_qk=576 and head_dim_v=512");
    TORCH_CHECK(mode==0||mode==3||mode==4,"mask_mode only supports 0, 3, 4");
    TORCH_CHECK(left>=-1 && left<=std::numeric_limits<int32_t>::max() &&
                    right>=-1 && right<=std::numeric_limits<int32_t>::max(),
                "win_left and win_right must be in [-1, INT32_MAX]");
    TORCH_CHECK(mode==4 || (left==-1 && right==-1),
                "win_left and win_right must be -1 for mask_mode 0 or 3");

    const int64_t batch=seq.numel();
    TORCH_CHECK(batch>0 && batch<65536,"batch_size (from cache_seqlens) must be in (0, 65536)");

    // Provided optional length tensors keep the pre-migration wrapper's
    // dtype/device/contiguity checks. Supported optional lengths are vectors:
    // a genuine empty vector (1-D, size 0) is the omitted/None-equivalent
    // form, while rank-0 or otherwise non-1-D supplied tensors are invalid and
    // rejected synchronously here instead of being passed to AICPU.
    auto checkProvided=[](const at::Tensor& t,const at::Tensor& seq,const char* name){
        TORCH_CHECK(t.device()==seq.device(),name," must be on the cache_seqlens device");
        TORCH_CHECK(t.scalar_type()==at::kInt,name," must be int32");
        TORCH_CHECK(t.is_contiguous(),name," must be contiguous");
    };
    if(cu) {
        checkProvided(*cu,seq,"cu_seqlens_q");
        TORCH_CHECK(cu->dim()==1,"cu_seqlens_q must be 1-D when provided");
    }
    if(used) {
        checkProvided(*used,seq,"seqused_q");
        TORCH_CHECK(used->dim()==1,"seqused_q must be 1-D when provided");
    }
    // Confirmed original-Host semantics: only a 1-D size-0 optional is
    // omitted. Rank-0/2-D empties are not valid length tensors (the original
    // path let them reach AICPU and fail); reject them synchronously here.
    auto present=[](const Optional& t){
        return t.has_value() && t->defined() && t->dim()==1 && t->size(0)>0;
    };
    const bool cuPresent=present(cu);
    const bool usedPresent=present(used);
    if(cuPresent && layout=="TND") {
        TORCH_CHECK(cu->numel()==batch+1,"cu_seqlens_q shape must be (batch_size+1,), but got ",cu->numel());
    }
    if(usedPresent) {
        TORCH_CHECK(used->numel()==batch,"seqused_q shape must be (batch_size,), but got ",used->numel());
    }
    if(layout=="TND") {
        TORCH_CHECK(cuPresent,"When layout_q is TND, cu_seqlens_q must be provided non-empty");
    } else {
        TORCH_CHECK(!cuPresent,"When layout_q is not TND, cu_seqlens_q must not be provided non-empty");
    }

    // The pre-migration torch wrapper sized its metadata allocation from the
    // platform totals; the original ACLNN host produced metadata content from
    // the current thread's stream resource limits when present, falling back to
    // the platform totals per resource. Preserve both behaviors.
    const auto platform=FlashMlaCoreNums();
    TORCH_CHECK(platform.first>0 && platform.second>0,"FlashMLA requires a 1:2 Cube/Vector platform");
    uint32_t limitedCube=0,limitedVector=0;
    const aclError cubeRc=aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE,&limitedCube);
    const aclError vectorRc=aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE,&limitedVector);
    const int64_t effectiveCube=cubeRc==ACL_SUCCESS?static_cast<int64_t>(limitedCube):platform.first;
    const int64_t effectiveVector=vectorRc==ACL_SUCCESS?static_cast<int64_t>(limitedVector):platform.second;
    TORCH_CHECK(effectiveCube>0 && effectiveVector>0,"effective Cube/Vector core counts must be positive");
    const int64_t words=(((platform.first+platform.second)*batch+1)*16+4095)/4096*4096;
    auto output=at::empty({words},seq.options());

    SglFlashMlaMetadataParam args;
    std::memset(&args,0,sizeof(args));
    args.length=sizeof(args);
    args.ioAddrNum=kSglFlashMlaMetadataIoCount;
    args.magic=kSglFlashMlaMetadataMagic;
    args.version=kSglFlashMlaMetadataVersion;
    args.cuSeqlensQ=static_cast<uint64_t>(cuPresent?reinterpret_cast<uintptr_t>(cu->data_ptr()):0U);
    args.cacheSeqlens=static_cast<uint64_t>(reinterpret_cast<uintptr_t>(seq.data_ptr()));
    args.sequsedQ=static_cast<uint64_t>(usedPresent?reinterpret_cast<uintptr_t>(used->data_ptr()):0U);
    args.metadata=static_cast<uint64_t>(reinterpret_cast<uintptr_t>(output.data_ptr()));
    args.metadataWords=output.numel();
    args.batchSize=static_cast<int32_t>(seq.numel());
    args.maxSeqlenQ=static_cast<int32_t>(maxq);
    args.maxSeqlenKv=static_cast<int32_t>(maxkv);
    args.numHeadsQ=static_cast<int32_t>(heads);
    args.numHeadsKv=static_cast<int32_t>(kvheads);
    args.headDimQk=static_cast<int32_t>(dqk);
    args.headDimV=static_cast<int32_t>(dv);
    args.maskMode=static_cast<int32_t>(mode);
    args.aicCoreNum=static_cast<int32_t>(effectiveCube);
    args.aivCoreNum=static_cast<int32_t>(effectiveVector);
    args.winLeft=left;
    args.winRight=right;
    std::memcpy(args.layoutQ,layout.c_str(),layout.size()+1);

    // The AICPU launcher is package-owned; the runtime copies the host args at
    // launch. Submit through torch-npu's OpCommand queue wrapper so this task
    // is ordered after previously queued torch writes on the same stream and
    // participates in NPUGraph capture. Captures below are by value so the
    // queued callback never reads stack lifetimes; the tensors keep their
    // storages alive until submission.
    auto& kernel=GetMetadataKernel(seq.device().index());
    auto stream=c10_npu::getCurrentNPUStream().stream(false);
    auto function=kernel.function;
    auto call=[args,function,stream,keepalive=output,seq_keep=seq,cu_keep=cu,used_keep=used]() mutable -> int {
        aclrtLaunchKernelAttr attr;
        std::memset(&attr,0,sizeof(attr));
        aclrtLaunchKernelCfg cfg={&attr,0};
        const aclError rc=aclrtLaunchKernelWithHostArgs(function,1U,stream,&cfg,&args,sizeof(args),nullptr,0);
        TORCH_CHECK(rc==ACL_SUCCESS,"FlashMLA metadata AICPU launch failed: ",rc);
        return rc;
    };
    at_npu::native::OpCommand::RunOpApi("sgl_flash_mla_metadata",call);
    return output;
}
at::Tensor flash_mla_metadata(const at::Tensor& seq,int64_t heads,int64_t kvheads,
    const Optional& cu,const Optional& used,const c10::optional<int64_t>& maxq,
    const c10::optional<int64_t>& maxkv,int64_t dqk,int64_t dv,
    const c10::optional<int64_t>& mode,int64_t left,int64_t right,
    const c10::optional<std::string>& layout) {
    return flash_mla_metadata_impl(seq,heads,kvheads,cu,used,maxq,maxkv,dqk,dv,mode,left,right,layout);
}
}
TORCH_LIBRARY_FRAGMENT(npu,m) {
 m.def("flash_mla_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, int head_dim_qk=576, int head_dim_v=512, int? mask_mode=0, int win_left=-1, int win_right=-1, str? layout_q='BSND') -> Tensor");
}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("flash_mla_metadata",TORCH_FN(sglang::npu_kernel::flash_mla_metadata));}
