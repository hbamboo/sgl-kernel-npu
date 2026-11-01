# Optional A5 companion. Loaded lazily only by the qualified SG opt-in route.
ascendc_library(moe_stable_routing_kernel STATIC op_kernel/stable_routing.cpp)
add_library(sgl_kernel_npu_moe_stable SHARED op_host/stable_torch.cpp)
target_include_directories(sgl_kernel_npu_moe_stable PRIVATE
    ${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment ${ASCEND_HOME_PATH}/include/experiment/metadef
    ${ASCEND_HOME_PATH}/include/experiment/runtime)
target_link_directories(sgl_kernel_npu_moe_stable PRIVATE
    ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(sgl_kernel_npu_moe_stable PRIVATE moe_stable_routing_kernel
    torch torch_cpu c10 torch_npu ascendcl tiling_api register platform ascendalog dl)
target_compile_options(sgl_kernel_npu_moe_stable PRIVATE -O3)
