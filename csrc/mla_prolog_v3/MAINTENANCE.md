# 维护指南

给"以后要动这个移植"的人。所有路径相对 `csrc/mla_prolog_v3/`。

## 1. 改动前必须知道的四件事

| # | 事实 | 为什么重要 |
|---|---|---|
| 1 | 三个投影权重必需 **bf16 + FRACTAL_NZ**（或 fp8+e8m0，由高层 API / shim 自动转换） | ND 权重会得到 cosine≈0.005 的"能跑但全错"结果 |
| 2 | `kvCacheStride0/krCacheStride0` 必须由 host 提供 | 缺了会让 PA block ≥1 的 token 全写错，T≤128 时看不出来 |
| 3 | tiling 从 `weightDkvKr` 的 N 维反推 `Hckv` | 改权重形状/走融合方案时必须同步改推导来源 |
| 4 | 工作区是 `[T, N]` 行主序，行跨度来自 `orgKc` | 任何改变 N 的改动都要同步 `orgKc` 与向量侧读取跨度 |

## 2. 常规流程

```bash
# 构建（全新目录最稳；增量构建需要 CANN merge_obj_text.sh 已打补丁）
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B /tmp/bk -S .
cmake --build /tmp/bk --target sgl_kernel_npu -j 16

# 验收（26 项，约 9 分钟；日志在 /tmp/mla_prolog_v3_acceptance）
bash csrc/mla_prolog_v3/tests/run_acceptance.sh <device>
```

改完代码的标准动作：**跑全量验收 → 更新 `ACCEPTANCE-SNAPSHOT.md`（HEAD、`.so` md5、耗时、结果）→ 提交**。

## 3. 调试入口

| 需求 | 开关 |
|---|---|
| tiling 失败时打印全部张量形状/类型 | `DSH_MLA_V3_DEBUG=1` |
| 只想跑到 launch 之前（测主机开销） | `MLA_PROLOG_V3_PROBE_TILING=1` |
| 强制核数 / mm1 瓦片 / 每步 token | `MLA_PROLOG_V3_CORES` / `MLA_PROLOG_V3_MM1_MIN_TILE` / `MLA_PROLOG_V3_MAX_STEP` |
| 单层真实权重 | `LAYER=<n> [PYBIND=1] python3 tests/test_real_glm_weights.py` |

## 4. 性能测量的三条纪律

1. **必须在卡空闲时测**：本机 8 张卡常被其它租户占用（90–95 GB、可达 100% 利用率），
   同一二进制在拥挤时慢约 10%，且会出现"33 µs / 48 µs"双稳态；
2. **同场 A/B**：对比两种配置要交替测量、取中位数，不要跨轮次比较；
3. **先看 kernel 再看总时长**：`tests/split_host_pybind.py` 能把主机侧（约 12 µs，其中 launch
   3.4 µs）与 kernel（约 18 µs）分开，避免把框架开销算进算子。

## 5. 已知未完成项

`带宽/算力利用率达到理论上限` 未达成（解码 29–30%、预填 13–16%）。原因见
`PERF-COUNTEREVIDENCE.md`，要突破请按 `DESIGN-kernel-rewrite.md` 的路线实施（方案 A 的
数值等价性已被 `tests/probe_fused_down.py` 证明，改动点已列全）。

## 6. 文件地图

| 文件 | 内容 |
|---|---|
| `README-port.md` | 实现细节、逐项移植改动、roofline 与 L1 证明 |
| `REPORT-glm47-mla-prolog.md` | 交付报告（TL;DR、适配原因、精度、性能、限制） |
| `PERF-COUNTEREVIDENCE.md` | 第 ③ 条的 12 组假设与实测反证 |
| `DESIGN-kernel-rewrite.md` | 三条改写路线（改动点/收益/验收门/回退条件） |
| `ACCEPTANCE-SNAPSHOT.md` | 最近一次干净构建 + 26 项验收的定格结果 |
| `tests/e2e/INTEGRATION.md` | 在自己的 SGLang 里怎么接 |
| `port_scripts/INDEX.md` | 91 个改动脚本的分类（已应用 / 实测回退 / 构建工具） |
