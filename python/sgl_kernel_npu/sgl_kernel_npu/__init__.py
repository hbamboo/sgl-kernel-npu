import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)
    for name in ("libsgl_kernel_npu_scatter.so", "libsgl_kernel_npu_scatter_mla.so"):
        extra_path = npu_path / "lib" / name
        if extra_path.is_file():
            torch.ops.load_library(str(extra_path))


_load_sgl_kernel_npu()
