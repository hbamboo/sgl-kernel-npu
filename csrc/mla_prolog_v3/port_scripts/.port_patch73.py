"""README: record the clean-build verification and the CANN script prerequisite."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

with io.open(P, encoding="utf-8") as f:
    s = f.read()

OLD = """## Acceptance runner"""
NEW = """## Clean-build verification (done from an empty build dir)

```
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \\
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \\
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \\
      -DBUILD_KERNELS_MODULE=ON -B /tmp/bk_clean -S .
cmake --build /tmp/bk_clean --target sgl_kernel_npu -j 16      # rc=0, 0 errors
```

Verified: configure rc=0, build rc=0, 0 `error:` lines, `libsgl_kernel_npu.so` (3.08 MB)
produced, and all 12 acceptance checks pass against that fresh binary.

**Build prerequisite (once per CANN installation)**: CANN's
`.../tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj_text.sh` links device
objects in place, which breaks on the second build (`unknown file type`).  Run
`port_scripts/port_fix_merge_script.sh` once (it keeps a `.orig` backup and makes the
merge idempotent); with it applied, incremental rebuilds work.  A fresh build directory
always works even without the patch.

## Acceptance runner"""
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)

P2 = os.path.join(ROOT, "csrc/mla_prolog_v3/REPORT-glm47-mla-prolog.md")
with io.open(P2, encoding="utf-8") as f:
    s = f.read()
OLD2 = """| 构建 | 纳入 sgl-kernel-npu 的 kernels 模块（A5 only），产出 `libsgl_kernel_npu.so` |"""
NEW2 = """| 构建 | 纳入 sgl-kernel-npu 的 kernels 模块（A5 only），产出 `libsgl_kernel_npu.so`；**空目录全新构建已验证**（configure/build 均 rc=0、0 error、产物 3.08 MB） |"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
OLD3 = """cmake --build build --target sgl_kernel_npu -j 16
```"""
NEW3 = """cmake --build build --target sgl_kernel_npu -j 16
```

> 一次性前置：CANN 的 `merge_obj_text.sh` 会原地合并 device object，第二次增量构建会报
> `unknown file type`。先执行一次 `csrc/mla_prolog_v3/port_scripts/port_fix_merge_script.sh`
> （保留 `.orig` 备份、使合并幂等）；全新 build 目录则无需该补丁。"""
assert s.count(OLD3) == 1
s = s.replace(OLD3, NEW3)
with io.open(P2, "w", encoding="utf-8") as f:
    f.write(s)
print("docs updated with clean-build evidence")
