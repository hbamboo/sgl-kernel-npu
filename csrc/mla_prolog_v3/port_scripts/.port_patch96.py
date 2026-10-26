#!/usr/bin/env python3
"""Add a 30-second TL;DR at the top of both documents."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

TLDR = """## TL;DR（30 秒了解交付状态）

| 目标条目 | 状态 | 一句话证据 |
|---|---|---|
| ① 完整复制 + 可正确编译 | ✅ | 空目录 configure/build 均 rc=0、0 error、228 s；`libsgl_kernel_npu.so` 3.08 MB |
| ② GLM-4.7 形状/量化适配 | ✅ | 放开 `Hcq=768`、MXFP8 自动反量化、6 类改动（详见第 3 节）|
| ② 参考 CPU golden 精度通过 | ✅ | **17/17 验收**：合成 golden（T=8/256）、31 个 token 边界、真实权重 4 层、PyBind 路径、与厂商逐位一致 |
| ③ 带宽/算力利用率达理论上限 | ❌ | 解码 29–30%、预填 13–16%；L1 512 KB 全部分配 + split-K 无候选 + 重叠无收益 + 主机侧到顶（第 7 节）|

**一条命令验收**：`bash csrc/mla_prolog_v3/tests/run_acceptance.sh <device>` → 期望 `passed=17 failed=0`（约 7 分钟）。

**性能**：解码 T=1 41.6 µs（`_out` 36.2 / PyBind **32.9**）；预填 T=256 89.7 µs、T=2048 649.9 µs。
**调用**：`torch.ops.npu.mla_prolog_v3` / `..._out` / PyBind 直调，三者输出逐位一致。
**SGLang**：算子已在真实 server 中被调用；端到端出词受上游 W8A8 int8 契约阻塞（`tests/e2e/README.md`）。
**要突破 ③**：需改写 kernel 数据流（第 12 节三条路线），需要独占卡与明确授权。

---

"""

P = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P, encoding="utf-8") as f:
    s = f.read()
marker = "> （SGLang 兼容层）。实现细节见同目录 `README-port.md`，端到端记录见 `tests/e2e/README.md`。\n\n---\n\n"
assert marker in s
s = s.replace(marker, marker + TLDR, 1)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
marker2 = "# MlaPrologV3 port notes (sgl-kernel-npu)\n\n"
assert marker2 in s
tldr_en = """## TL;DR

Ported from `ops-transformer/attention/mla_prolog_v3` and adapted for GLM-4.7-Flash
(`Hcq=768`, `N=20`, `D=192`, MXFP8).  Status: builds from an empty tree (228 s, 0 errors),
**17/17 acceptance checks pass** (`tests/run_acceptance.sh`, ~7 min), decode 32.9 us via
the PyBind entry / 36.2 us via `torch.ops..._out`, prefill 89.7 us at T=256 and 649.9 us at
T=2048.  The bandwidth/compute-utilisation target (decode 29-30%, prefill 13-16%) is **not**
met: the kernel is L1-capacity bound (512 KB fully allocated) and every non-rewrite path was
measured and excluded - see "Remaining prefill headroom" and the fusion/split-K notes below.
Full report: `REPORT-glm47-mla-prolog.md`.

"""
s = s.replace(marker2, marker2 + tldr_en, 1)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("TL;DR added to both documents")
