# 交付说明（结案版）

> 目标原文：「将 mla_prolog_v3 完整复制到 sgl-kernel-npu 并确保可正确编译；基于 GLM-4.7-Flash 与
> MlaPrologV3 形状/量化不匹配的原因完成适配，参考 CPU golden 确保精度验证通过；进行性能优化，
> 确保带宽和算力利用率达到理论上限（精度优先）」。
>
> 结案状态：**① ② 达成；③ 未达成**——已用 20 组实测把"不改 kernel 结构"的空间穷尽，并给出
> 形状上限的定量证明与改写施工图。用户于第 66 轮决定「以当前状态结案」。

## 1. 最终验收基线（本轮实测，HEAD `caecd85`）

| 项 | 值 |
|---|---|
| 构建 | 全新目录 `/tmp/bk_close`，**238 s**、**0 error** |
| 产物 | `python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so`，md5 `9f81f830e487db4da2e86d467ccb1fad` |
| 验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5` → **549 s，27/27 通过** |
| 工作树 | 干净 |

## 2. 三条目标的结论

### ① 完整复制 + 可正确编译 —— 达成

- `csrc/mla_prolog_v3/`（算子本体）+ `csrc/mla_prolog/`（依赖的 tiling 与 arch35 kernel），
  并把 `mla_prolog_v3_pybind.cpp` 接入 `csrc/CMakeLists.txt`；
- 空目录 238 s 构建、0 error；`libsgl_kernel_npu.so` 内同时提供 `torch.ops.npu.*` 与 PyBind 绑定；
- 增量构建另需 CANN `merge_obj_text.sh` 幂等补丁（已归档 `port_scripts/`）。

### ② GLM-4.7-Flash 适配 + CPU golden 精度 —— 达成

原算子对 GLM-4.7-Flash 不匹配的三处，及处理方式：

| 不匹配 | 原因 | 处理 |
|---|---|---|
| `Hcq=768` 被拒 | 内置校验只允许 1536/2048 | 放开 `supportedHcqSize`（保留原值仍合法）|
| `N=20` 不在允许集 | 校验按 2 的幂枚举 | 放开到 16 的倍数并补 D=192 的分块分支 |
| MXFP8 权重（fp8 + e8m0） | 算子只认 bf16/int8 或特定 fp8 排布 | 反量化到 bf16（`weight_quant_mode=0`），并在 Python 层做成自动适配 |

**精度证据（27 项验收，全部可一键复跑）**：

- CPU golden（T=8）：五路 cosine `0.999994 / 0.999996 / 0.999998 / 0.999998 / 0.999998`；
- **31 个 token 边界**（1…1000，覆盖所有内部分块边界）全部 ≥ 0.9995；
- **4 层真实 MXFP8 权重**（layer 0/2/23/46）五路 ≥ 0.9999949；
- **与 CANN 内置算子逐位一致**：4 组几何（Hcq 1536/2048 × N 8/16/32/128 × D 128/192），五路 max\|d\| = **0**；
- 工程项：双 stream 并发、ACL graph 捕获+重放、输出缓冲区边界校验、shape 交替、纯逻辑单测、三路径逐位一致。

### ③ 利用率达理论上限 —— **未达成**（有完整举证）

现状：解码 kernel 29–30%（18.2 µs vs 5.3 µs roofline）、预填 13–16%（T=2048 时 54.5 TFLOP/s）。

**为什么达不到（每条都有实测）**：

1. **硬件峰值是真的**：大 GEMM 4096³ 达 423.5 TFLOP/s（100%）、4 GiB copy 达 3318 GB/s（101%）
   ⇒ 不是"峰值虚标"，而是**形状**问题；
2. **解码是 GEMV 形状**：该形状的实际上限约 1 TB/s（厂商 `torch.mv` 只有 384 GB/s，我们已达 ≈0.95 TB/s）；
   且解码耗时**与权重字节数无关**（3.75→16.5 MB 时延不变，斜率 0.02 µs/MB）——瓶颈是 4 段串行 +
   5 次跨核 barrier 的固定开销；
3. **预填被 m 限死**：m=128 时同形状 `torch.matmul` 也只有 54.4 TFLOP/s（与我们相同），
   但 m=512 能到 159.6、m=2048 到 335 ⇒ 机会在"每步 token 数"，而它被向量段
   `ceil(step/64 AIV 核) ≤ 2` 卡住（超一即 507015，已定位到向量段逐 token 逻辑，非容量问题）；
4. **另 40% 是权重重复流**：T=2048 时 16 步 × 17.3 MB = 277 MB，按 0.95 TB/s ≈ 290 µs / 653 µs。

**已实测排除的 20 组路线**：核数 8–32、mm1 瓦片 32–192、每步 token 64–128、mm3/mm1/mm2 baseK、
split-K/N、阶段重叠、mm2 扩核、并发下投影、主机三条路径与缓存、acgraph、vector-only GEMV、
每步 token 上限、融合下投影（数值已证逐位等价）等。

**要突破的唯一路径**：把每步 token 从 128 提到 512，需重构向量段的逐 token 循环（同时把权重流量 ÷4、
m 效率 ×3）。施工图见 `DESIGN-kernel-rewrite.md`。

## 3. 性能（安静卡实测）

| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| µs | 41.0 | 41.8 | 52.8 | 89.8 | 169.8 | 329.2 | 649.7 |
| TFLOP/s | — | 3.31 | 21.0 | 49.3 | 52.2 | 53.8 | **54.5** |

调用路径（单次延迟，T=1）：分配版 41.6 µs → `_out` 36.2（−13%）→ PyBind **32.9**（−21%）。
**稳态吞吐**（47 层背靠背 = 一个 decode step）：18.6 µs/层，合计 **0.87 ms/step**；此时三条入口无差别。

## 4. 交付物清单

| 路径 | 内容 |
|---|---|
| `csrc/mla_prolog_v3/op_host/` | torch 入口（分配版 / `_out` 版）、PyBind 绑定、tiling 缓存、输出校验、`DSH_MLA_V3_DEBUG`/`DSH_MLA_V3_DUMP` 诊断 |
| `csrc/mla_prolog_v3/op_kernel/` | 单入口 kernel（运行期按 tilingKey 分发） |
| `csrc/mla_prolog/` | 依赖的 tiling 与 arch35 kernel（仅放开 Hcq/N/步长约束与 D=192 分支） |
| `python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3.py` | 高层 API：`mla_prolog_v3` / `mla_prolog_v3_out` / `allocate_outputs`（自动反量化、转向、NZ、模式推断、输出分配） |
| `python/.../mla_prolog_v3_compat.py` | SGLang 兼容层：`custom::npu_mla_prolog_v3` 别名、路由补丁、ND→NZ、MXFP8→bf16 |
| `csrc/mla_prolog_v3/tests/` | 27 项验收 + 12 个诊断/基准脚本 |
| `csrc/mla_prolog_v3/*.md` | 7 份文档（见下） |

## 5. 文档索引

| 文档 | 用途 |
|---|---|
| `README-port.md` | 实现细节、逐项移植改动、L1 与 roofline 分析 |
| `REPORT-glm47-mla-prolog.md` | 交付报告（中文，TL;DR + 12 节） |
| `ACCEPTANCE-SNAPSHOT.md` | 验收基线（HEAD、md5、耗时、逐项结果） |
| `PERF-COUNTEREVIDENCE.md` | 第 ③ 条的 20 组反证与硬件标定 |
| `DESIGN-kernel-rewrite.md` | 三条改写路线（改动点/收益/验收门/回退条件） |
| `MAINTENANCE.md` | 维护指南（四个必知事实、构建/验收流程、性能测量纪律） |
| `tests/e2e/INTEGRATION.md` | 在自己的 SGLang 里怎么接 |

## 6. 复现方式

```bash
# 构建
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B /tmp/bk -S .
cmake --build /tmp/bk --target sgl_kernel_npu -j 16

# 验收（期望 passed=27 failed=0）
bash csrc/mla_prolog_v3/tests/run_acceptance.sh <device>

# 性能
python3 csrc/mla_prolog_v3/tests/bench_mla_prolog.py
python3 csrc/mla_prolog_v3/tests/bench_layer_loop.py     # 稳态吞吐
```

## 7. 已知限制

1. 第 ③ 条未达成（见 §2 ③），突破需重构向量段以放大每步 token 数；
2. 原生 MXFP8 `weight_quant_mode=3` 在 Hcq=768 下 mm3 的 scale 映射不正确，生产路径用 bf16 反量化；
3. SGLang 的 MLAPO 融合链路要求 W8A8 int8，MXFP8 checkpoint 在 prefill 会走 int8 分支并因缺
   `input_offset` 报错——这是上游契约，已在 `tests/e2e/INTEGRATION.md` 记录；
4. 本机 8 张卡常被其它租户共用，性能对比必须同窗口 A/B（实测差异可达 ±20%）。
