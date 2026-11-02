# Dedicated companion SO for the unified-576 KV RMSNorm+RoPE cache update.
#
# Built as a separate shared object (like libsgl_kernel_npu_scatter_mla.so) so it
# can be loaded by absolute path in an isolated process without rebuilding the
# main sgl_kernel_npu extension. Compiles only for Ascend950-series SoCs via the
# parent csrc/CMakeLists.txt A5 gate.

ascendc_library(kv_rmsnorm_unified_kernel STATIC op_kernel/kv_rmsnorm_unified_cache.cpp)
ascendc_include_directories(kv_rmsnorm_unified_kernel PRIVATE op_kernel)

add_library(sgl_kernel_npu_kv_rmsnorm_unified SHARED op_host/kv_rmsnorm_unified_torch.cpp)
target_link_libraries(sgl_kernel_npu_kv_rmsnorm_unified PRIVATE kv_rmsnorm_unified_kernel)
target_include_directories(sgl_kernel_npu_kv_rmsnorm_unified PRIVATE
    op_host op_kernel
    ${PROJECT_OP_SRC_BASE}/utils
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/experiment
    ${ASCEND_INCLUDE_DIR}/experiment/metadef
    ${ASCEND_INCLUDE_DIR}/experiment/runtime)
target_link_directories(sgl_kernel_npu_kv_rmsnorm_unified PRIVATE
    ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(sgl_kernel_npu_kv_rmsnorm_unified PRIVATE
    torch torch_cpu c10 torch_npu ascendcl tiling_api register platform ascendalog dl)
