"""Docs: record the PyBind11 fast path."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """| 1 | 41.6 us (36.3 with `_out`) | 18.2 us | 5.3 us (BW) | 29% |"""
NEW = """| 1 | 41.6 us (36.3 `_out`, **32.9 PyBind**) | 18.2 us | 5.3 us (BW) | 29% |"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,"""
NEW2 = """* **PyBind11 fast path**: `import libsgl_kernel_npu as _C; _C.mla_prolog_v3_out(...)`
  skips the torch dispatcher: T=1 36.7 -> **32.9 us** (-10.4%), T=8 35.4 -> 32.8 us,
  bit-identical results.  The binding lives in the same `libsgl_kernel_npu.so`
  (`mla_prolog_v3_pybind.cpp`), and `mla_prolog_v3_compat.direct()` returns it.
  The caller must pass bf16 FRACTAL_NZ weights - the dtype/format adaptation only
  happens on the torch.ops path.
* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD3 = """| 调用 | `torch.ops.npu.mla_prolog_v3(...)`（分配输出）/ `torch.ops.npu.mla_prolog_v3_out(...)`（调用方给缓冲区，解码快 13–14%）|"""
NEW3 = """| 调用 | 三条路径：`torch.ops.npu.mla_prolog_v3`（分配输出）/ `torch.ops.npu.mla_prolog_v3_out`（调用方给缓冲区，解码快 13–14%）/ **PyBind 直调** `libsgl_kernel_npu.mla_prolog_v3_out`（跳过 dispatcher，再快 10%，T=1 = 32.9 µs）|"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
OLD4 = """优化轨迹（T=1）：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希 key）"""
NEW4 = """优化轨迹（T=1）：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希 key）
→ 41.6（核数启发式）→ 36.3（`_out` 变体）→ **32.9 µs（PyBind 直调）**；
旧轨迹续："""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated")
