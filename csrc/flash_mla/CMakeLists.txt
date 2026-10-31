# FlashMLA is shipped with the sgl_kernel_npu wheel as a companion operator
# library. Keeping it separate avoids rebuilding unrelated kernels and preserves
# the existing libsgl_kernel_npu.so ABI.
if(NOT SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    return()
endif()

set(FLASH_MLA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB FLASH_MLA_GENERAL_SRCS CONFIGURE_DEPENDS
    ${FLASH_MLA_ROOT}/op_kernel/sgl_flash_mla_[01]_[0124].cpp)
list(LENGTH FLASH_MLA_GENERAL_SRCS FLASH_MLA_GENERAL_COUNT)
if(NOT FLASH_MLA_GENERAL_COUNT EQUAL 8)
    message(FATAL_ERROR "FlashMLA requires all eight query-layout/dtype entries")
endif()

ascendc_library(flash_mla_direct_kernel STATIC
    ${FLASH_MLA_GENERAL_SRCS})
ascendc_include_directories(flash_mla_direct_kernel PRIVATE
    ${FLASH_MLA_ROOT}/vendor/common
    ${FLASH_MLA_ROOT}/vendor/common/op_kernel/arch35
    ${ASCEND_INCLUDE_DIR})
ascendc_compile_definitions(flash_mla_direct_kernel PRIVATE -DHAVE_TILING -DHAVE_WORKSPACE)
ascendc_compile_options(flash_mla_direct_kernel PRIVATE --cce-auto-sync=off)

add_library(sgl_kernel_npu_flash_mla SHARED
    ${FLASH_MLA_ROOT}/flash_mla_torch.cpp
    ${FLASH_MLA_ROOT}/flash_mla_metadata.cpp)
target_include_directories(sgl_kernel_npu_flash_mla PRIVATE
    ${FLASH_MLA_ROOT}
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
    flash_mla_direct_kernel torch_npu torch torch_cpu c10
    ascendcl tiling_api register platform ascendalog dl)
