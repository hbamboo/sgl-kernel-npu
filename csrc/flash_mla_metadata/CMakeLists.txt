# Package-owned FlashMLA metadata companion. The host operator is an OBJECT
# library assembled into the existing FlashMLA wheel host library (no extra
# cross-DSO C++ ABI); the AICPU producer remains a separate package-owned .so.
if(NOT SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    return()
endif()

set(FLASH_MLA_METADATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(FLASH_MLA_METADATA_AICPU_DIR ${FLASH_MLA_METADATA_ROOT}/aicpu)

add_library(sgl_kernel_npu_flash_mla_metadata_host OBJECT
    ${FLASH_MLA_METADATA_ROOT}/flash_mla_metadata.cpp)
target_include_directories(sgl_kernel_npu_flash_mla_metadata_host PRIVATE
    ${FLASH_MLA_METADATA_ROOT}
    ${PROJECT_OP_SRC_BASE}/utils
    ${ASCEND_INCLUDE_DIR}
    ${ASCEND_INCLUDE_DIR}/external
    ${ASCEND_INCLUDE_DIR}/experiment/runtime
    ${ASCEND_INCLUDE_DIR}/experiment/platform
    ${TORCH_DIR}/include
    ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include)

# AICPU producer: built with the CANN aarch64 AICPU toolchain, loaded at
# runtime through ACL CPU-kernel mode. No OPP registration is involved.
find_program(SGL_FLASH_MLA_AICPU_CXX
    NAMES aarch64-target-linux-gnu-g++
    PATHS ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc/bin
          ${ASCEND_HOME_PATH}/tools/hcc/bin
    NO_DEFAULT_PATH)
if(NOT SGL_FLASH_MLA_AICPU_CXX)
    message(FATAL_ERROR "FlashMLA metadata requires the CANN aarch64 AICPU compiler "
                        "(aarch64-target-linux-gnu-g++) under ${ASCEND_HOME_PATH}")
endif()

set(FLASH_MLA_AICPU_SOURCES
    ${FLASH_MLA_METADATA_AICPU_DIR}/metadata_kernel.cpp)
set(FLASH_MLA_AICPU_HEADERS
    ${FLASH_MLA_METADATA_AICPU_DIR}/metadata_param.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/flash_mla_metadata.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/load_balance/load_balance_common.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/load_balance/base_info.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/load_balance/section_stream_k/section_stream_k.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/load_balance/section_stream_k/section_stream_k_def.h
    ${FLASH_MLA_METADATA_AICPU_DIR}/load_balance/section_stream_k/section_stream_k_impl.h
    ${FLASH_MLA_METADATA_ROOT}/../flash_mla/op_kernel/flash_mla_shared.h)
set(FLASH_MLA_AICPU_SO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsgl_flash_mla_metadata_aicpu.so)
set(FLASH_MLA_AICPU_JSON ${FLASH_MLA_METADATA_AICPU_DIR}/libsgl_flash_mla_metadata_aicpu.json)
set(FLASH_MLA_AICPU_JSON_OUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsgl_flash_mla_metadata_aicpu.json)
add_custom_command(
    OUTPUT ${FLASH_MLA_AICPU_SO}
    COMMAND ${SGL_FLASH_MLA_AICPU_CXX}
            -shared -fPIC -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -fno-common
            -I${FLASH_MLA_METADATA_AICPU_DIR}
            -o ${FLASH_MLA_AICPU_SO}
            ${FLASH_MLA_AICPU_SOURCES}
    DEPENDS ${FLASH_MLA_AICPU_SOURCES} ${FLASH_MLA_AICPU_HEADERS}
    COMMENT "Building package-owned FlashMLA metadata AICPU library"
    VERBATIM)
add_custom_command(
    OUTPUT ${FLASH_MLA_AICPU_JSON_OUT}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FLASH_MLA_AICPU_JSON} ${FLASH_MLA_AICPU_JSON_OUT}
    DEPENDS ${FLASH_MLA_AICPU_JSON}
    COMMENT "Staging package-owned FlashMLA metadata AICPU descriptor"
    VERBATIM)
add_custom_target(sgl_flash_mla_metadata_aicpu ALL
    DEPENDS ${FLASH_MLA_AICPU_SO} ${FLASH_MLA_AICPU_JSON_OUT})
add_dependencies(sgl_kernel_npu_flash_mla_metadata_host sgl_flash_mla_metadata_aicpu)

install(FILES ${FLASH_MLA_AICPU_SO} ${FLASH_MLA_AICPU_JSON_OUT} DESTINATION lib)
