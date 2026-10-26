#!/usr/bin/env python3
"""Report: before/after steady-state profile of the kernel."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## 6. 为什么没有到 roofline（定量结论）"""
NEW = """## 6. kernel 内部：优化前后的稳态 profile

msprof `PipeUtilization`，同一脚本、同一设备，取中位数：

| 指标 | 优化前（32 核，首版） | **优化后（16 核，当前）** |
|---|---|---|
| kernel 时长 (T=1) | 28.96 µs | **18.64 µs（−36%）** |
| AIC MAC（cube） | 9.5% | **29.3%（约 3x）** |
| AIC MTE2（HBM→L1） | 35.9% | 48.9% |
| AIC MTE1（L1→L0） | 15.3% | 51.2% |
| AIC scalar | 30.7% | 36.2% |
| AIV scalar | — | 32.9% |

预热填（T=256，32 核）：MTE1 **56.4%** 为最忙管道、MTE2 28.4%、MAC 17.9%、scalar 10%
——即预填阶段是典型的 **L1→L0 搬运受限**，与第 7 节"L1 全部分配"的结论互相印证。

## 7. 为什么没有到 roofline（定量结论）"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

# renumber the following sections (7 -> 8, 8 -> 9, 9 -> 10, 10 -> 11)
for old_no, new_no in (("## 7. 已知限制", "## 8. 已知限制"),
                       ("## 8. 复现", "## 9. 复现"),
                       ("## 9. 一页式验收清单", "## 10. 一页式验收清单"),
                       ("## 10. 后续路线（若要把利用率推上去）", "## 11. 后续路线（若要把利用率推上去）")):
    assert s.count(old_no) == 1, old_no
    s = s.replace(old_no, new_no)

with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("report: profile section added, sections renumbered")
