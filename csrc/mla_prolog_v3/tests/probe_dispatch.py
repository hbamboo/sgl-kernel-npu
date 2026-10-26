"""See how torch invokes the library impl (positional vs keyword)."""
import sys

import torch
import torch_npu

sys.path.insert(0, "/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu")

lib = torch.library.Library("probe", "FRAGMENT")
lib.define(
    "op(Tensor a, Tensor b, *, int mode=0, float eps=1e-5, str? s=None) -> (Tensor)")

seen = {}


def _impl(*args, **kwargs):
    seen["args"] = [type(a).__name__ for a in args]
    seen["kwargs"] = list(kwargs)
    return args[0] + args[1]


lib.impl("op", _impl, "PrivateUse1")

dev = "npu:0"
x = torch.ones(4, device=dev, dtype=torch.bfloat16)
y = torch.ones(4, device=dev, dtype=torch.bfloat16)
out = torch.ops.probe.op(x, y, mode=7, s="k")
print("args seen   :", seen["args"])
print("kwargs seen :", seen["kwargs"])
print("out ok      :", bool((out == 2).all()))
