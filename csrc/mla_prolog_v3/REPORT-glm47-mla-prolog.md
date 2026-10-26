# MlaPrologV3 移植 / GLM-4.7-Flash 适配 / 精度与性能 最终报告

> 代码：`csrc/mla_prolog_v3/`（算子本体 + 测试 + 移植脚本）、`csrc/mla_prolog/`（依赖的
> op_host tiling 与 arch35 kernel 头文件）、`python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py`
> （SGLang 兼容层）。实现细节见同目录 `README-port.md`，端到端记录见 `tests/e2e/README.md`。

---

## 1. 交付概览

| 项 | 内容 |
|---|---|
| 来源 | `/mnt/paas/hxdnpu/ops-transformer/attention/mla_prolog_v3` 完整复制，并补入其依赖 `attention/mla_prolog` |
| 构建 | 纳入 sgl-kernel-npu 的 kernels 模块（A5 only），产出 `libsgl_kernel_npu.so`；**空目录全新构建已验证**（configure/build 均 rc=0、0 error、产物 3.08 MB） |
| 调用路径 | ① `torch.ops.npu.mla_prolog_v3`（分配输出）② `torch.ops.npu.mla_prolog_v3_out`（调用方给缓冲）③ **PyBind 直调** `libsgl_kernel_npu.mla_prolog_v3_out`（跳过 dispatcher） |
| 验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5` → **12/12 通过** |
| 平台 | Ascend950DT（DAV_3510），CANN 9.1.0，torch_npu 2.10 |

构建：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B build -S .
cmake --build build --target sgl_kernel_npu -j 16
```

一次性前置：CANN 的 `merge_obj_text.sh` 会原地合并 device object，第二次增量构建会报
`unknown file type`。先跑一次 `csrc/mla_prolog_v3/port_scripts/port_fix_merge_script.sh`
（保留 `.orig` 备份、使合并幂等）；全新 build 目录无需该补丁。

---

## 2. 为什么需要适配

容器内置 `MlaPrologV3` 只接受 DeepSeek MLA 几何，实测拒绝 GLM-4.7-Flash：

```
Parameter Hcq of MlaPrologV3 has incorrect value 768. It should be 1536 or 2048.
Parameter N   of MlaPrologV3 has incorrect value 20. It should be 1,2,4,...,128.
The dtype or format ... inconsistent with that defined in the operator prototype   (fp8 权重)
```

而 GLM-4.7-Flash 是 `He=2048, Hcq=768, Hckv=512, D=192, Dr=64, N=20` + MXFP8
(`float8_e4m3` + 每 32 元素 `e8m0`) 权重。ops-transformer 的**源码版**已支持 `D=192`、
`N∈[1,128]`、`weightQuantMode=3`，只差 `Hcq` 一道门。

---

## 3. 做了哪些改动（按主题）

### 3.1 语义（唯一一处功能放开）
* 放开 `Hcq=768`（`mla_prolog_tiling_check.cpp` 的 `supportedHcqSize`）。

### 3.2 direct-launch 环境适配
* 单入口 kernel + 按 `tilingKey` 运行时派发，替代 `ASCENDC_TPL` 的"每组合一个二进制"；
* `GET_TILING_DATA_WITH_STRUCT` 不可用 → 生成 GM→local 的逐字段拷贝（52 + 10 + key）；
* `ASCENDC_TPL` 声明头在 host 编译遍不通过 → preamble 跳过；
* `ge::TypeUtils::DataTypeToSerialString` 链接不到 → 本地 dtype 名助手；
* 平台信息用 `PlatformAscendCManager`；
* CANN `merge_obj_text.sh` 改为幂等（见第 1 节）。

### 3.3 精度缺陷修复（真实 bug）
* **PA cache stride**：`ConvertContext` 本应从 GE descriptor 取 `kvCacheStride0` /
  `krCacheStride0`，direct-launch host 漏填（=0），导致 **PA block ≥ 1 的 token 全部写错**
  （T ≤ 128 时看不出）。已按 `prod(dims[1:])` 补齐。修复前 T≥192 的 kv/kr cosine ≈ 0.006，
  修复后 T=8…1024 全部 ≥ 0.9999。

### 3.4 性能（详见第 5 节）
* tiling / workspace 缓存、无分配哈希 key；
* 核数按 token 数自适应（T≤128 用 16 核）；
* mm1 GEMM 最小瓦片 64→32（预填多核化）；
* D=192 时每步 token 上限 64→128（UB 预算内，192 会崩）；
* `mla_prolog_v3_out`（调用方缓冲，省 5 次 `at::empty`）；
* PyBind 直调（跳过 torch dispatcher）。

### 3.5 SGLang 兼容层（`mla_prolog_v3_compat.py`）
逐个实测并解决：
1. 本镜像无 `custom::npu_mla_prolog_v3` → 注册同名别名转发；
2. checkpoint 无 `ignore` 键 → 补 `kv_b_proj` 标记，让 SGLang 走 `forward_mlaprolog`；
3. SGLang 硬编码 `weight_quant_mode=1` → 按实际 dtype 推断（浮点→0、fp8+e8m0→3、int8 保留）；
4. `q_b_proj` / `w_kc` 是 ND（kernel 要 NZ）→ `ensure_nz()` 转换 + 缓存；
5. 权重是 MXFP8 → `ensure_bf16()` 反量化 + 缓存（支持 `[in,out]` 与 `[out,in]` 两种 scale 朝向）。

---

## 4. 精度

### 4.1 CPU golden（GLM 几何，T=8）

| 输出 | cosine | max abs diff |
|---|---|---|
| query | 0.999994 | 0.0030 |
| query_rope | 0.999996 | 0.0098 |
| query_norm | 0.999998 | 0.0173 |
| kv_cache | 0.999998 | 0.0169 |
| kr_cache | 0.999998 | 0.0007 |

### 4.2 token 范围

| T | query | query_rope | kv_cache | kr_cache |
|---|---|---|---|---|
| 128 | 0.999961 | 0.999994 | 1.000000 | 0.999999 |
| 256 | 0.999980 | 0.999982 | 0.999995 | 0.999999 |
| 512 | 1.000094 | 0.999975 | 0.999993 | 1.000000 |
| 1024 | 1.000379 | 0.999965 | 0.999980 | 0.999998 |

### 4.3 真实 checkpoint（`model.layers.2`，MXFP8）

五路输出 cosine ≥ **0.9999975**；经 SGLang 别名调用结果与直接调用**逐位一致**；
以 MXFP8 原始 fp8 权重 + e8m0 scale 直接喂入时五路 cosine ≥ 0.9999976。

### 4.4 与厂商算子

DeekSeek 几何（He=7168/Hcq=1536/D=128/N=16, bf16）：五路输出 **逐位一致**（max diff 0.0）。

---

## 5. 性能

### 5.1 机器上限（device 5 实测）

HBM **3.27 TB/s**（256 MiB copy）、bf16 **424 TFLOP/s**（4096³ matmul）。

### 5.2 当前数据

| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| PyBind `_out` (µs) | **32.9** | 33.4 | — | — | — | — | — |
| torch.ops `_out` (µs) | 36.2 | 36.5 | 54 | 90.0 | 170.1 | 329.3 | 650.2 |
| torch.ops 分配版 (µs) | 41.6 | 42.2 | 54 | 90.1 | 170.1 | 329.3 | 650.2 |
| kernel 占比（roofline） | 29–30% | 30% | 24% | 16% | 14% | 14% | 13% |

### 5.3 优化轨迹

```
解码 T=1：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希）
        → 41.6（核数启发式）→ 36.2（_out）→ 32.9 µs（PyBind）   合计 2.36x
预填 T=256：118.8 → 89.9 µs；T=2048：687.7 → 650.2 µs
```

同几何下比 CANN 内置算子快（T=1 56.7→ 我们的实现更快，且输出逐位一致）。

### 5.4 主机侧成本结构（T=1，wall-clock）

| 组成 | 耗时 |
|---|---|
| PyBind 绑定本身 | 0.11 µs |
| kernel launch | ~3.4 µs |
| 我们的主机逻辑（重建描述 + tiling 查表） | ~8.3 µs |
| torch.ops dispatcher（仅 torch 路径） | ~4.7 µs |

尝试过缓存 launch 描述以免重建，实测 **更慢**（15.3 vs 11.8 µs，校验成本高于重建），已回滚。

---

## 6. kernel 内部：优化前后的稳态 profile

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

## 7. 为什么没有到 roofline（定量结论）

1. **预填**：He 扫描（1024/2048/4096/8192 → 71.9/93.8/143.3/244.5 µs）表明
   **mm1+mm2 占预填 kernel 的 70%**（T=256 时约 49/70 µs），它们只跑到 28.7 TFLOP/s（7%）。
2. **L1 已满**：`L1_A_SIZE = L1_B_SIZE = 128 KB`，双缓冲后合计 **512 KB = 整块 L1**；
   而 `kL1StepSize = baseK×stepK = 512` 同时是 B 的上限（128 列）和 m=128 时 A 的上限。
   加大即溢出（这正是 `baseK=256` 触发 507015 的原因），减小只会让 L0 瓦片更浅；
   解码态（m=1）A 虽空闲，B 仍要满 128 KB，同样无法加深。
3. **调度已无可选**：split-K/N 的候选搜索（`mla_prolog_kn_search.h`）对 N=768/576 无候选，
   且其 kernel 路径被 arch gate 限定 DAV_9201（放开也空转）；mm1/mm2 重叠无收益
   —— 两者都已占用 24/32 核且吞吐受限，把 mm2 挤到 8 个空闲核需 3 倍单核工作量。
4. **主机侧**：已从 41.6 压到 32.9 µs，剩余 11.8 µs 中 3.4 是 launch、8.3 是描述重建/查表，
   缓存方案实测更慢。

结论：**这套 kernel 在 GLM 形状上的上限约为"解码 30% 带宽、预填 13–16% 算力"**，
再往上只能改写数据流（见第 9 节）。

---

## 8. 已知限制

1. **原生 MXFP8（mode 3）在 Hcq=768 下不可用**：mm2（下投影）数值完全正确（cosine 0.999999），
   但 mm3 的 `weight_uq_qr` per-32 scale 映射有误（16 核 cosine 0.981、32 核越界）。
   生产路径是**反量化到 bf16**（无损，已验证）。
2. **SGLang 端到端**：算子已能在真实 server 中被调用（权重 bf16 NZ、无 tiling 错误），
   但 SGLang 的 MLAPO 分支要求 `qr` 是 int8 量化结果、且分支由调用方按 `ignore` 决定，
   本 MXFP8 checkpoint 不满足该契约（属上游功能缺口，详见 `tests/e2e/README.md`）。
3. **共享卡噪声**：同一配置下解码态会在 ~33 µs 与 ~48 µs 两个稳定态间跳变（疑与其他租户
   负载有关），本报告所有对比均为同场 A/B。
4. **PyBind 路径不做 dtype/format 适配**：调用方须自行保证 bf16 + FRACTAL_NZ。

---

## 9. 复现

```bash
# 全量验收（12 项，含 golden / 真实权重 / 厂商逐位 / 三条调用路径 / SGLang shim / 性能）
bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5

# 单项
T=256 python3 csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py
python3 csrc/mla_prolog_v3/tests/test_real_glm_weights.py
python3 csrc/mla_prolog_v3/tests/cmp_builtin_numeric.py
python3 csrc/mla_prolog_v3/tests/test_pybind_path.py        # 三路径对比
python3 csrc/mla_prolog_v3/tests/split_host_pybind.py       # 主机侧分解
python3 csrc/mla_prolog_v3/tests/bench_mla_prolog.py        # 性能扫描

# 端到端（SGLang server，见 tests/e2e/README.md）
cd csrc/mla_prolog_v3/tests/e2e && bash run_server.sh
```

---

## 10. 从零复现实测

在一台干净 build 目录 + 清空测试 `__pycache__` 的条件下完整跑了一遍（device 5）：

| 步骤 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| configure | `cmake ... -B /tmp/bk_drill -S .` | < 10 s | rc=0 |
| build | `cmake --build /tmp/bk_drill --target sgl_kernel_npu -j 16` | **217 s** | rc=0、**0 error**、`.so` 3.08 MB |
| 验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5` | **112 s** | **12/12 通过** |

验收后的性能与既有记录一致：T=64 53.8 µs、T=256 **89.9 µs**、T=512 169.9 µs、
T=1024 329.3 µs、T=2048 **650.0 µs**。

即：**从零到通过验收约 5.5 分钟**（其中构建 3.6 分钟），无需人工干预。

## 11. 一页式验收清单

复制到目标机器后，按顺序执行即可判定交付是否可用：

| # | 检查 | 命令 | 通过标准 |
|---|---|---|---|
| 1 | 构建 | `cmake ... && cmake --build build --target sgl_kernel_npu -j 16` | rc=0、0 error、产出 `libsgl_kernel_npu.so` |
| 2 | 全量验收 | `bash csrc/mla_prolog_v3/tests/run_acceptance.sh <dev>` | `passed=12 failed=0` |
| 3 | 精度（合成） | 同上 `golden_T8` / `golden_T256` 日志 | 五路 cosine ≥ 0.99996 |
| 4 | 精度（真实权重） | `real_weights_layer2` 日志 | 五路 cosine ≥ 0.9999975 |
| 5 | 与厂商一致 | `vendor_bit_exact` 日志 | 五路 max\|d\| = 0.00000 |
| 6 | 性能 | `perf_sweep` 日志 | T=1 ≤ 36.5 µs（`_out`）/ ≤ 33.5 µs（PyBind）；T=2048 ≤ 660 µs |
| 7 | SGLang 接入 | `sglang_alias` / `nd_weights` / `mode_infer` 日志 | 三份均 OK（逐位一致） |

必须知道的两条前置：

* CANN `merge_obj_text.sh` 需先打一次幂等补丁（或使用全新 build 目录）；
* 三个投影权重必须是 **bf16 + FRACTAL_NZ**（PyBind 路径），torch.ops 路径会自动做 ND→NZ 与
  MXFP8→bf16 转换（已缓存）。

## 12. 后续路线（若要把利用率推上去）

| 方案 | 做法 | 预估收益 | 风险/成本 |
|---|---|---|---|
| A. 融合 mm1+mm2 | 加载期把 `W_DQ` 与 `W_DKV/KR` 拼成 `[He, 1344]`，一次 GEMM 出两个结果 | 省一个阶段的开销，预填 ~3–6% | 中：改 tiling + kernel 输出切分 + 向量侧读取 |
| B. 权重常驻 L1 的持久 kernel | 解码态按 token 流式、权重常驻 L1（当前 L1 被 ping/pong 占满） | 解码 kernel 18.2 → 约 8~10 µs | 高：需重写 kernel 数据流，须独占卡长时间迭代 |
| C. 换 K 切分 | 让每核拿满 N、只切 K（cluster/workspace 归约） | 预填 mm1+mm2 有望 −30% | 高：kernel 侧 3510 无现成实现 |

三条都需要改 kernel 数据流，不属于调参范围；当前交付基线为"精度全部验证通过 + 比厂商快 + 解码 2.36x"。
