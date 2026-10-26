#!/usr/bin/env python3
"""Consistency pass: refresh the stale numbers/status in README-port.md and add a
one-page acceptance checklist to the final report."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """Progression of the decode path: 77.7 us (first working version) -> 48.5 us (tiling +
workspace cache) -> 45.1 us (hash keys) -> **41.6 us** (core-count heuristic), i.e."""
NEW = """Progression of the decode path: 77.7 us (first working version) -> 48.5 us (tiling +
workspace cache) -> 45.1 us (hash keys) -> 41.6 us (core-count heuristic) -> 36.2 us
(`_out`) -> **32.9 us** (PyBind), i.e."""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """Last run: 11/11 pass, decode 36.1 us (`_out`), T=2048 650 us."""
NEW2 = """Last run: **12/12 pass**, decode 36.2 us (`_out`) / 32.9 us (PyBind), T=256 89.8 us,
T=2048 649.9 us."""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)

OLD3 = """| 2048 | 650.9 us | 627.5 us | 83.6 us (compute) | 13% |"""
NEW3 = """| 2048 | 649.9 us | 626.5 us | 83.6 us (compute) | 13% |"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD4 = """## 9. 后续路线（若要把利用率推上去）"""
NEW4 = """## 9. 一页式验收清单

复制到目标机器后，按顺序执行即可判定交付是否可用：

| # | 检查 | 命令 | 通过标准 |
|---|---|---|---|
| 1 | 构建 | `cmake ... && cmake --build build --target sgl_kernel_npu -j 16` | rc=0、0 error、产出 `libsgl_kernel_npu.so` |
| 2 | 全量验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh <dev>` | `passed=12 failed=0` |
| 3 | 精度（合成） | 同上 `golden_T8` / `golden_T256` 日志 | 五路 cosine ≥ 0.99996 |
| 4 | 精度（真实权重） | `real_weights_layer2` 日志 | 五路 cosine ≥ 0.9999975 |
| 5 | 与厂商一致 | `vendor_bit_exact` 日志 | 五路 max\\|d\\| = 0.00000 |
| 6 | 性能 | `perf_sweep` 日志 | T=1 ≤ 36.5 µs（`_out`）/ ≤ 33.5 µs（PyBind）；T=2048 ≤ 660 µs |
| 7 | SGLang 接入 | `sglang_alias` / `nd_weights` / `mode_infer` 日志 | 三份均 OK（逐位一致） |

必须知道的两条前置：

* CANN `merge_obj_text.sh` 需先打一次幂等补丁（或使用全新 build 目录）；
* 三个投影权重必须是 **bf16 + FRACTAL_NZ**（PyBind 路径），torch.ops 路径会自动做 ND→NZ 与
  MXFP8→bf16 转换（已缓存）。

## 10. 后续路线（若要把利用率推上去）"""
assert s.count(OLD4) == 1
s = s.replace(OLD4, NEW4)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs refreshed")
