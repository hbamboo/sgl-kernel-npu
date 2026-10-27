#!/usr/bin/env python3
"""Report: add the steady-state serving numbers (the counter-evidence doc is already updated)."""
import io

P = "/mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md"
s = io.open(P, encoding="utf-8").read()

OLD = """| 调用路径 | ① `torch.ops.npu.mla_prolog_v3`（分配输出）② `torch.ops.npu.mla_prolog_v3_out`（调用方给缓冲）③ **PyBind 直调** `libsgl_kernel_npu.mla_prolog_v3_out`（跳过 dispatcher）；三条路径对同一输入输出**逐位一致**，且都已用真实权重对过 golden |"""
NEW = """| 调用路径 | ① `torch.ops.npu.mla_prolog_v3`（分配输出）② `torch.ops.npu.mla_prolog_v3_out`（调用方给缓冲）③ **PyBind 直调** `libsgl_kernel_npu.mla_prolog_v3_out`（跳过 dispatcher）；三条路径对同一输入输出**逐位一致**，且都已用真实权重对过 golden |
| 稳态吞吐 | 47 层背靠背（一次 decode step）：**18.6 µs/层**（T=1）→ **0.87 ms/step**；T=64 时 31.1 µs → 1.46 ms。此时三条入口无差别，成本由 kernel 决定（主机优化只降单次延迟）|"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
io.open(P, "w", encoding="utf-8").write(s)
print("report updated")
