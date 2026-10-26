#!/usr/bin/env python3
"""Report: record the measured from-scratch reproduction (build 217 s, acceptance 112 s)."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## 10. 一页式验收清单"""
NEW = """## 10. 从零复现实测

在一台干净 build 目录 + 清空测试 `__pycache__` 的条件下完整跑了一遍（device 5）：

| 步骤 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| configure | `cmake ... -B /tmp/bk_drill -S .` | < 10 s | rc=0 |
| build | `cmake --build /tmp/bk_drill --target sgl_kernel_npu -j 16` | **217 s** | rc=0、**0 error**、`.so` 3.08 MB |
| 验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5` | **112 s** | **12/12 通过** |

验收后的性能与既有记录一致：T=64 53.8 µs、T=256 **89.9 µs**、T=512 169.9 µs、
T=1024 329.3 µs、T=2048 **650.0 µs**。

即：**从零到通过验收约 5.5 分钟**（其中构建 3.6 分钟），无需人工干预。

## 11. 一页式验收清单"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD2 = """## 11. 后续路线（若要把利用率推上去）"""
NEW2 = """## 12. 后续路线（若要把利用率推上去）"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD3 = """Verified: configure rc=0, build rc=0, 0 `error:` lines, `libsgl_kernel_npu.so` (3.08 MB)
produced, and all 12 acceptance checks pass against that fresh binary."""
NEW3 = """Verified: configure rc=0, build rc=0, 0 `error:` lines, `libsgl_kernel_npu.so` (3.08 MB)
produced, and all 12 acceptance checks pass against that fresh binary.

Reproduction drill (clean build dir + cleared ``__pycache__``): configure < 10 s,
**build 217 s** (3.6 min, -j 16), **acceptance 112 s** (12/12) - about 5.5 minutes from
scratch to green with no manual steps."""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("reproduction record added")
