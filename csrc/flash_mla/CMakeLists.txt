# FlashMLA is shipped with the sgl_kernel_npu wheel as a companion operator
# library. Keeping it separate avoids rebuilding unrelated kernels and preserves
# the existing libsgl_kernel_npu.so ABI.
if(NOT SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    return()
endif()

set(FLASH_MLA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(FLASH_MLA_KERNEL_SRC ${FLASH_MLA_ROOT}/op_kernel/flash_mla.cpp)

# One upstream-shaped template entry (op_kernel/flash_mla.cpp) is the only kernel
# source. ASCENDC_TPL_SEL in op_kernel/arch35/flash_mla_template_tiling_key.h is
# the selection source of truth; direct ascendc_library builds do not run the OPP
# tiling-key expander, so the entry carries the matching explicit specializations.
# ORIG_DTYPE_Q is a compile-time configuration in upstream too; the bf16/fp16
# variants need distinct C entry names only because both device binaries are
# linked into the single wheel host DSO. The bf16 entry keeps the upstream name.
ascendc_library(flash_mla_bf16_kernel STATIC ${FLASH_MLA_KERNEL_SRC})
ascendc_include_directories(flash_mla_bf16_kernel PRIVATE
    ${FLASH_MLA_ROOT}/op_kernel
    ${FLASH_MLA_ROOT}/op_kernel/common
    ${FLASH_MLA_ROOT}/op_kernel/common/arch35
    ${ASCEND_INCLUDE_DIR})
ascendc_compile_definitions(flash_mla_bf16_kernel PRIVATE
    -DHAVE_TILING
    -DHAVE_WORKSPACE
    -DORIG_DTYPE_Q=DT_BF16)
ascendc_compile_options(flash_mla_bf16_kernel PRIVATE --cce-auto-sync=off)

ascendc_library(flash_mla_fp16_kernel STATIC ${FLASH_MLA_KERNEL_SRC})
ascendc_include_directories(flash_mla_fp16_kernel PRIVATE
    ${FLASH_MLA_ROOT}/op_kernel
    ${FLASH_MLA_ROOT}/op_kernel/common
    ${FLASH_MLA_ROOT}/op_kernel/common/arch35
    ${ASCEND_INCLUDE_DIR})
ascendc_compile_definitions(flash_mla_fp16_kernel PRIVATE
    -DHAVE_TILING
    -DHAVE_WORKSPACE
    -DORIG_DTYPE_Q=DT_FLOAT16
    -DFLASH_MLA_ENTRY=flash_mla_fp16)
ascendc_compile_options(flash_mla_fp16_kernel PRIVATE --cce-auto-sync=off)

add_library(sgl_kernel_npu_flash_mla SHARED
    $<TARGET_OBJECTS:sgl_kernel_npu_flash_mla_metadata_host>
    ${FLASH_MLA_ROOT}/op_host/flash_mla_torch.cpp)
target_include_directories(sgl_kernel_npu_flash_mla PRIVATE
    ${FLASH_MLA_ROOT}/op_host
    ${FLASH_MLA_ROOT}/op_kernel
    ${PROJECT_OP_SRC_BASE}/utils
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/external
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
    ${ASCEND_INCLUDE_DIR}/experiment/platform
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include)
target_link_directories(sgl_kernel_npu_flash_mla PRIVATE
    ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib)
target_link_libraries(sgl_kernel_npu_flash_mla PRIVATE
    flash_mla_bf16_kernel
    flash_mla_fp16_kernel
    torch_npu torch torch_cpu c10
    ascendcl tiling_api register platform ascendalog dl)
install(TARGETS sgl_kernel_npu_flash_mla LIBRARY DESTINATION lib)
