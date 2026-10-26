#!/usr/bin/env python3
"""Port patch #2: drop the GE TypeUtils::DataTypeToSerialString dependency.

The CANN 9.1.0 layout ships that helper with the pre-cxx11 ABI only, which the
torch extension (cxx11 ABI) cannot link against.  The helper is only used to
render error messages, so a tiny local mapping is enough.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "csrc/mla_prolog/op_host/mla_prolog_tiling_check.cpp")

HELPER = """namespace optiling {

// [sgl-kernel-npu port] local dtype-name helper: the GE TypeUtils variant is not
// linkable with the cxx11 ABI used by the torch extension.
inline std::string MlaPrologDataTypeToStr(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT: return "DT_FLOAT";
        case ge::DT_FLOAT16: return "DT_FLOAT16";
        case ge::DT_BF16: return "DT_BFLOAT16";
        case ge::DT_INT8: return "DT_INT8";
        case ge::DT_INT16: return "DT_INT16";
        case ge::DT_INT32: return "DT_INT32";
        case ge::DT_INT64: return "DT_INT64";
        case ge::DT_UINT8: return "DT_UINT8";
        case ge::DT_FLOAT8_E4M3FN: return "DT_FLOAT8_E4M3FN";
        case ge::DT_FLOAT8_E8M0: return "DT_FLOAT8_E8M0";
        case ge::DT_HIFLOAT8: return "DT_HIFLOAT8";
        default: return "DT_UNKNOWN";
    }
}
"""

with io.open(PATH, encoding="utf-8") as f:
    src = f.read()

assert src.count("TypeUtils::DataTypeToSerialString") == 6, src.count("TypeUtils::DataTypeToSerialString")
src = src.replace("TypeUtils::DataTypeToSerialString", "MlaPrologDataTypeToStr")

# inject the helper right after `namespace optiling {` (first occurrence)
anchor = "namespace optiling {\n"
assert src.count(anchor) == 1
src = src.replace(anchor, HELPER)

with io.open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("patched", PATH)
