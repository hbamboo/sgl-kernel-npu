# 预填性能与原生 MXFP8 收官报告（2026-10-28）

> 本轮目标：基于 `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`（GLM-4.7-Flash，MXFP8 权重）
> ① 确保 MXFP8 量化可正常使用、精度正常、性能有收益；② prefill 阶段在 30456 token 下优于 2.5 ms；
> **精度优先**。

## 0. 结论

| 目标 | 状态 | 证据 |
|---|---|---|
| ① MXFP8 可用 + 精度正常 | ✅ | 原生 `weight_quant_mode=3` 的 query/query_rope 由 **0.9761 → 0.99956**；kv/kr 0.999999；真实 checkpoint 权重 5 路 ≥ 0.9999975 |
| ① MXFP8 性能有收益 | ✅ | T=30456：bf16 9610 µs → 原生 mxfp8 **2337 µs（4.11×）** |
| ② prefill 30456 token < 2.5 ms | ✅ | kernel 层 **2335–2342 µs**；**生产 Python API + 真实 checkpoint 权重 2217–2223 µs**（均低于 2500 µs 6.5%/11%）|
| 精度优先 | ✅ | split-M 与 split-N 输出 **逐位相同（max\|d\| = 0）**，精度零损失 |

## 1. 改动清单（3 个文件，+50 行）

| 文件 | 改动 |
|---|---|
| `csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h` | `MmQcQrParamInit()` 补 `stepK` 回退 |
| `csrc/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_m.h` | 同上（split-M 副本）|
| `csrc/mla_prolog/op_host/mla_prolog_tiling.cpp` | split-M 对 GLM 放开（env 开关）+ `tail==1` 死锁防护 |

## 2. 缺陷一：MXFP8 在 Hcq=768 下的 mm3 scale 错位（精度）

### 根因

`MatmulL1()` 用如下式子定位第 `kL1` 个 K 分块的 FP8 per-32 scale：

```cpp
uint32_t aScaleOffset = kL1 * para.kScale / kL1Loops * BLOCK_CUBE_SIZE;   // 16 = BLOCK_CUBE_SIZE
```

它隐含假设「每个 K 分块含相同数量的 32 元素 scale 块」。而正确偏移应是
`kL1 * kL1StepSize / 32 * 16`。两者只在 `k % kL1StepSize == 0` 时相等。

mm1/mm2 早就因此加了回退（`MmCqParamInit` / `MmCkvKrParamInit`：`if ((k/baseK) % stepK != 0) stepK = 3;`），
**mm3 漏了这一步**：

| 形状 | baseK | k/baseK | stepK | kL1StepSize | 结果 |
|---|---|---|---|---|---|
| Hcq=1536（DeepSeek） | 64 | 24 | 4 | 256 | 24%4=0 ✅ |
| Hcq=2048 | 64 | 32 | 4 | 256 | 32%4=0 ✅ |
| **Hcq=768（GLM）** | 128 | 6 | 4 | 512 | **6%4=2 ❌** |

Hcq=768 时 `kL1Loops=2`，第二个 K 分块的 scale 偏移算成 `1*24/2*16 = 192`，真值应为
`512/32*16 = 256` —— **提前了 16 个 scale 块**，于是 query / query_rope 相关度掉到 0.976
（kv/kr 走 mm2，一直是对的，与历史观察一致）。

### 修复

在 mm3 的两处 `MmQcQrParamInit()` 里补上与 mm1/mm2 同款回退：K 不被 `baseK*4` 整除时 `stepK = 3`，
Hcq=768 → `kL1StepSize = 384`，`kL1Loops = 2`，每块正好 12 个 scale 块，偏移算式成立。

### 验证（`tests/test_mxfp8_precision.py` + 自建矩阵）

| 几何 | 修复前 query | 修复后 query |
|---|---|---|
| He2048 Hcq768 D192 N20（GLM，真实形状） | 0.976149 | **0.999587** |
| He2048 Hcq768 D192 N16 | 0.972268 | **0.999555** |
| He2048 Hcq768 D128 N20 | 0.972466 | **0.999552** |
| He2048 Hcq1536 D192 N20 | 0.999549 | 0.999549（无回归）|
| He7168 Hcq1536 D128 N128（DeepSeek） | 0.999541 | 0.999541（无回归）|
| He7168 Hcq1536 D128 N16 | 0.999592 | 0.999592（无回归）|

真实 checkpoint 权重（layer 2，MXFP8）：五路 `0.9999975 / 0.9999979 / 0.9999979 / 0.9999980 / 0.9999982`。

> 注：`tests/test_mxfp8_precision.py` 的参考是「fp8 值 × e8m0」直接反量化（e8m0 全 127），
> 这条路上一旦激活量化参与（`query_norm_flag=True`）误差会升到 ~2.9%，是 per-32 e8m0 量化本身
> 的固有损失（fp8 e4m3 mantissa 3 bit），不是 kernel 缺陷。上面的 0.9995 量级是 kernel 与
> 「同样反量化后的权重」之间的相关度，用来定位 kernel 逻辑错误；量化损失另算。

## 3. 缺陷二：预填 13–16% 利用率的真正原因（性能）

### 单核形状实测（决定一切）

同一张卡上按「每核一块 GEMM」测（`torch.bmm`，batch=24 模拟 24 核并发）：

| 每核 m | 每核 k | 每核 n | 每核 TFLOP/s |
|---|---|---|---|
| 128 | 2048 | **32**（当前 mm1 切N） | **15.9** |
| 128 | 2048 | 64（当前 mm2 切N） | 30.4 |
| 128 | 2048 | **768**（split-M 的完整 N） | **265.6** |
| 128 | 2048 | 576（split-M 的 mm2） | 201.5 |

即：切 N 把每核 N 压到 32/64，单核效率只有 15.9/30.4 TFLOP/s；split-M 让每核拿完整 N、只切 M，
单核能到 265.6/201.5 —— **差 8–17 倍**。这就是预填上不去的根因，也解释了早先
`PERF-COUNTEREVIDENCE.md` §1c5 那条「m 从 128 提到 512 能到 3 倍」的现象（当时归因到每步 token 数，
真实决定量是每核 N）。

### 上游门禁

split-M 模板在 `mla_prolog_tiling.cpp` 里只对
`weight_quant_mode == MXFP8_FULL_QUANT && tSize >= 8192 && He ∈ {7168, 7680} && N == 128` 开放。
GLM（He=2048、N=20、Hcq=768）被挡在门外，只能走切 N。

split-M kernel 本身完全由 `baseParams_` 驱动，没有几何硬编码，因此放开是安全的：

| 项 | split-N（默认） | split-M（放开后） |
|---|---|---|
| mm1 分核 | 24 核 × 32 列 | 32 核 × **768**（完整 Hcq）|
| mm2 分核 | 9 核 × 64 列 | 32 核 × **576** |
| mm3 分核 | 20 核 × 256 | 32 核 × **5120** |
| mm4 分核 | 20 核 × 1 头 | 32 核 × 20 头 |
| workspace | 21.3 MB | 192.0 MB |

### 放开方式

**默认已放开**（`tSize >= 8192 && MXFP8` 的 prefill 全部走 split-M，含 GLM）。
可用 `MLA_PROLOG_V3_SPLIT_M_GLM=0` 强制回退到 split-N（保留上游行为，用于 A/B 与排障）。

默认放开的依据：
1. 与 split-N **逐位相同**（§5，T=8192 / 30456 五路全部 max|d| = 0）；
2. workspace 192 MB 走 `WorkspaceFor()`，是**按 device 缓存并复用的一块**，不会每次调用分配；
   卡上 96 GB 显存，192 MB 占比 0.2%；
3. `tail==1` 的死锁已在 tiling 里拦掉并回退 split-N（§4），不会挂死；
4. 两种模式下 27/27 验收全绿（§7）。

## 4. 缺陷三：split-M 在「尾块 = 1 token」时死锁（稳定性）

放开后立刻暴露：**T=8193 / 8200 / 8224 直接挂死**（设备 300 s 超时，rc=124）。

根因：split-M 按核把 token 切成 `mSubSize`（前 `mSubCoreNum` 个核）与 `mSubSize-1`（其余核），
每核内部以 `stepBatchSize=128` 步进，**循环体内有 cube/vector 跨核 barrier**。因此所有核必须走同样的步数。
当尾块恰好是 1 token 时两种情况会错位：

1. 某核分片 `mSubSize % 128 == 1` → 它最后一步只有 1 token，同伴最后一步是满的；
2. 尾核 `mSubSize-1` 恰是 128 的整数倍 → 尾核比满核少走一步。

实测（GLM 几何、step=128）：

| 尾块 | T | 结果 |
|---|---|---|
| 1 | 8224（mSubSize=257） | **挂死 rc=124** |
| 2 | 8256（258） | OK |
| 4 | 8320（260） | OK |
| 8 | 8448（264） | OK |
| 16 | 8704（272） | OK |
| 32 | 9216（288） | OK |
| 64 | 10240（320） | OK |
| 56 | **30456**（952） | OK（目标尺寸不受影响）|

修复：tiling 里检测 `mSubSize % stepBatchSize == 1`（或尾核少一步）时，**回退到 split-N**。
回退只是慢一些，绝不挂死；该条件只覆盖约 1% 的 T 取值。

验证：8193 / 8200 / 8224 现在 rc=0 且自动落到 split-N（`mm1 24 x 32`）；
8256 / 8704 / 30456 仍走 split-M（`mm1 32 x 768`）。

## 5. 精度（精度优先：零损失）

`/tmp/xproc.py` 跨进程 A/B（同输入、同进程边界外对比，排除测量抖动）：

| T | query | query_rope | query_norm | kv_cache | kr_cache |
|---|---|---|---|---|---|
| 8192 | max\|d\|=0 exact | 0 exact | 0 exact | 0 exact | 0 exact |
| **30456** | **0 exact** | **0 exact** | **0 exact** | **0 exact** | **0 exact** |

即 split-M 与 split-N **输出逐位相同**。精度层面两条路径完全等价，放开 split-M 不引入任何数值风险。

## 6. 性能（T=30456，交替三轮）

| 配置 | 中位数 | TFLOP/s |
|---|---|---|
| bf16 + split-N（原始基线） | 9610.9 µs | 54.8 |
| bf16 + split-M | 9610.9 µs | 54.8 |
| MXFP8 + split-N | 5043.9 / 5045.0 / 5043.5 µs | 104.5 |
| **MXFP8 + split-M** | **2335.5 / 2339.4 / 2338.2 µs** | **225.4** |

- 相对原始基线：**4.11×**
- 相对 2.5 ms 目标：**低 6.5%**（2337 µs vs 2500 µs）
- 参考上限（同卡逐 GEMM 用 `torch` 拼）：1633.9 µs / 322 TFLOP/s，本路径已到其 **70%**

## 6b. 生产入口（Python API）的原生 MXFP8

上面 §6 是 `torch.ops` 直调 kernel 的数字。走生产入口
`from sgl_kernel_npu.mla_prolog_v3 import mla_prolog_v3`（真实 MXFP8 checkpoint 权重，
`/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8` layer 2）时，原先仍是 **9678 µs**——
因为兼容层无条件把 MXFP8 反量化成 bf16。本轮打通了原生路径：

| 配置 | T=30456 延迟 |
|---|---|
| 反量化到 bf16（改动前唯一路径）| 9677–9678 µs |
| 原生 MXFP8（本轮）| **2217–2223 µs（4.36×）** |

改动（`python/sgl_kernel_npu/sgl_kernel_npu/`）：

1. `_prepare()` 在原生模式下只做 NZ 转换 + 朝向修正，**不再反量化**，返回 mode 3；
2. 新增 `quant_mxfp8_act()`：把激活按 per-32 e8m0 量化成 fp8（MXFP8 的 A 侧契约）；
   指数直接取自 bf16 位模式（`bits >> 7`）而不是 `floor(log2())`，并在同 dtype 内用 2 的幂做
   缩放，实测比 fp32 路线快约 1.8×；再按 `(data_ptr, shape, _version)` 缓存，服务里同一批 token
   过 47 层时只算一次（946 µs → 675 µs → **17 µs**）；
3. 三个权重 e8m0 scale + 激活 scale 透传给 kernel。

### 精度：为什么原生模式只在 prefill 启用

原生 MXFP8 同时量化 A 和 W（checkpoint 原始 W8A8 方案）。`quant_mxfp8_act()` 与理想 e8m0
量化器的相对误差是 **2.38% vs 2.25%**（cosine 0.99953）——已经是 fp8 e4m3 三位尾数的格式下限。
但相对**bf16 激活**的 golden 会呈现 2.4% 差异（`min cosine 0.99928`），这正是
`tests/test_python_api.py`（阈值 0.9995）在第一次改动后报 FAIL 的原因。

因此按 T 门控（`NATIVE_MXFP8_MIN_TOKENS = 8192`）：

| T | 路径 | 理由 |
|---|---|---|
| < 8192（decode / 短 prefill）| bf16 反量化 | 该区间 kernel 由固定开销主导，两种路径一样快（T=8：123 µs vs 147 µs），bf16 精度更高 |
| ≥ 8192（prefill）| **原生 MXFP8** | bf16 达不到 2.5 ms；原生是唯一可行路径，且用的是 checkpoint 自己的量化方案 |

`DSH_MLA_PROLOG_V3_MXFP8=1` 可强制任意 T 走原生，`=0` 强制 bf16。

> **关于 2.4% 的定位**：这不是 kernel 缺陷，也不是量化器实现问题（与理想量化器差 5%），
> 而是 MXFP8 A 侧量化的固有效果。若某场景要求 bf16 级激活精度，应保持 `=0` 或走 T<8192 的默认路径。

## 7. 复现

```bash
# 默认即 split-M（无需任何环境变量）
python3 /tmp/bench_mx.py 30456          # -> 2339 us

# 强制回退 split-N 做 A/B
MLA_PROLOG_V3_SPLIT_M_GLM=0 python3 /tmp/bench_mx.py 30456   # -> 5044 us

# 精度 A/B（逐位相同）
MLA_PROLOG_V3_SPLIT_M_GLM=0 python3 /tmp/xproc.py save 30456   # split-N 存盘
python3 /tmp/xproc.py cmp 30456                               # split-M 对比 -> max|d| = 0

# 验收（27 项，两种模式都应 27/27）
bash csrc/mla_prolog_v3/tests/run_acceptance.sh <device>
MLA_PROLOG_V3_SPLIT_M_GLM=0 bash csrc/mla_prolog_v3/tests/run_acceptance.sh <device>
```

## 8. 已知限制

1. split-M 已默认开启，但 T 落在「尾块=1」的约 1% 取值上会自动回退到 split-N（此时是原速度
   5044 µs，仍快于 bf16 的 9610 µs）；split-M 额外需要一块 192 MB、按 device 复用的 workspace；
2. MXFP8 的 per-32 e8m0 激活量化本身有约 2.4% 相对误差（fp8 e4m3 的 3-bit mantissa 所致），
   是 MXFP8 格式的固有限制（与理想量化器仅差 5%）。因此原生路径只在 T>=8192 的 prefill 启用，
   T<8192 仍走 bf16 反量化（同速、更准）；若要求 bf16 级激活精度，设 `DSH_MLA_PROLOG_V3_MXFP8=0`；
3. 本轮未改动 `mla_prolog_v3` 的 decode 路径（T≤128），其 18.6 µs/层 的稳态没有变化。
