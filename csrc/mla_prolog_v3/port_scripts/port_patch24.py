#!/usr/bin/env python3
"""Fix: provide the PA cache stride0 values to the tiling context.

`MlaPrologTiling::ConvertContext` normally derives `kvCacheStride0/krCacheStride0`
from the GE tensor stride descriptor (contiguous stride of dim 0 in elements).  The
direct-launch host never set them, so the kernel's cache scatter used stride 0:
correct as long as all tokens fit in the first PA block (T <= blockSize), silently
wrong beyond that.
"""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/op_host/mla_prolog_v3_host.cpp")

OLD = """    size_t workSpaces[2] = {0U, 0U};
    ctx.workSpaces = workSpaces;"""
NEW = """    // [sgl-kernel-npu port] PA cache stride of dim 0 in elements, i.e. what
    // ConvertContext would read from the GE tensor descriptor.  The kernel uses it to
    // address PA blocks, so a zero stride silently breaks every token that spills into
    // block >= 1.
    auto stride0 = [](const std::vector<int64_t> &dims) -> uint64_t {
        if (dims.empty()) {
            return 0U;
        }
        uint64_t s = 1U;
        for (size_t i = 1U; i < dims.size(); ++i) {
            s *= static_cast<uint64_t>(dims[i]);
        }
        return s;
    };
    ctx.kvCacheStride0 = stride0(tensors.kvCache);
    ctx.krCacheStride0 = stride0(tensors.krCache);

    size_t workSpaces[2] = {0U, 0U};
    ctx.workSpaces = workSpaces;"""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("cache stride fix applied")
