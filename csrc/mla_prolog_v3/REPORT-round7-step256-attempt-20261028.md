# 第 7 轮：按 step=256 重推三处 tile 分配（失败记录）

> 基线 `1ad5d4d`（tag `baseline-2.3ms` = `60ddbad`）。**代码零改动**，实验已完整回退，
> 工作树干净、二进制已还原。
> 结论：**按推导的 recipe 实现后仍然崩溃，且失败与 tile 尺寸无关。**

## 1. 三处 tile 的完整推导（这部分是对的，有复用价值）

从源码取约束（不是猜）：

| 资源 | 容量 | 占用公式 | 出处 |
|---|---|---|---|
| L0A | 32 KB/缓冲 | `align(step,16) × baseK × esz` | `MmadParams(mSize,n,baseK)` @ service_matmul:623,761,946 |
| L0B | 32 KB/缓冲 | `baseN × baseK × esz` | 同上 |
| L0C | 64 KB/缓冲 | `align(step,16) × baseN × 4` | `GetTensorC` fixpipe 输出 `mSize × nSize` |
| L1_A | 128 KB/缓冲 | `step × baseK × stepK × esz` | `LoadL1A` → `CopyNDGmToL1(A[kL1*kL1StepSize], m, kL1StepSize, k)` |
| L1_B | 128 KB/缓冲 | `baseK × stepK × baseN × esz` | `LoadL1BOuter` |

**关键结论（第 6 轮漏掉、本轮补上）：`baseN` 是必须同时下调的变量。**
L0C 只含 `baseN`、不含 `baseK`，所以**只降 baseK 永远救不了 L0C** —— 这解释了此前所有 baseK-only 实验为何全崩。

`step=256` 时 `L0C` 要求 `baseN ≤ 65536/(256×4) = 64`（upstream mm3/mm4 是 128，正好超一倍）。

推导出的完整可行组合：

| GEMM | baseK | stepK | baseN | A_L0 | C_L0 | A_L1 |
|---|---|---|---|---|---|---|
| mm1/mm2 (fp8,K=2048) | 128 | 4 | 64 | 32768 | 65536 | 131072 |
| mm3 (fp8,K=1536) | 64 | 6 | 64 | 16384 | 65536 | 98304 |
| mm4 (bf16,K=128) | 32 | 4 | 64 | 16384 | 65536 | 65536 |

## 2. 实现后仍然崩溃

把上表实现在 `MmCq/MmCkvKr/MmQcQr/MmQn ParamInit` 里，用 `MLA_PROLOG_V3_STEP256=1` 门控
（dump 已确认 `step=256` 生效、workspace 从 20.9 MB 增到 25.1 MB）。仍然：
`errorStr: The address for CUBE to operate L0A is out of bounds`。

### 决定性实验一：逐个 GEMM 放大 step

给四个 GEMM 各自的 step 开关，每次只让一个用 256、其余保持 128：

| 只放大的 GEMM | 结果 |
|---|---|
| mm1 | FAILED |
| mm2 | FAILED |
| mm3 | FAILED |
| mm4 | FAILED |
| 全部 | FAILED |

**单个 GEMM 也是崩的** ⇒ 与某个 GEMM 的 tile 参数无关。

### 决定性实验二：找出精确崩溃点

在 recipe tile 下扫 step：

| step | 结果 |
|---|---|
| 128 | OK |
| **129** | **FAILED** |
| 132 / 144 / 160 / 192 / 224 / 256 | FAILED |

**step=129（只多 1 个 token）就崩**。129 的 tile 只比 128 大不到 1%，
所以这**不是容量问题**，是某个按 step 的整数假设被打破。

### 决定性实验三：修向量核数

怀疑是 `vectorBlockNum_ = min(step, aivNum)`（64）导致 step=129 时每核要装 3 个 token。
改成 `vectorBlockNum_ = aivNum_` 后仍然 FAILED。

## 3. 已排除的原因清单

| 假设 | 排除依据 |
|---|---|
| L0A 容量不足 | A tile 压到 16384（一半余量）仍崩 |
| L0C 容量不足 | baseN 降到 64（正好 65536）仍崩 |
| L1_A 容量不足 | 压到 65536（一半余量）仍崩 |
| 某个 GEMM 特有 | 逐个 GEMM 放大均崩 |
| workspace 不一致 | dump 显示随 step 正确增长 |
| 向量核数/每核 token | 改成 aivNum_ 仍崩 |
| 是容量问题 | step=129 只大 1% 就崩 |

## 4. 剩下的线索

DFX 寄存器：`su error info: 0xfffffeb905c25972`，`cube error info: 0x7ca800000000`，
`l1 error info: 0x7c970000000000`，PC 偏移 `0x9444`（与 step=128 正常时同一段代码）。

`su` 是 scalar unit —— 这暗示**标量侧**的某个地址计算在 step>128 时溢出，
而不是 cube 的 tile 越界。可能的方向：
- 某个 `uint8_t`/`uint16_t` 的中间量在 step>128 时回绕；
- `SetLoadDataRepeatWithStride({0,1,0,dstStride})` 的 `dstStride = CeilDiv(mSize,16)`
  在 mSize=144 时是 9，可能超出某处硬件字段宽度。

**但这些都是猜测，本轮没有验证。** 我没有继续，因为：
- 已连续 6 次"推导→实现→崩溃"；
- 每次崩溃都是 507015 设备级异常，会打断同卡进程；
- 继续盲试的边际收益很低。

## 5. 结论

| 项 | 状态 |
|---|---|
| 三处 tile 的推导 | ✅ 完成，公式与可行组合已给出（本文 §1，可复用） |
| step=256 实现 | ❌ 失败，且失败与 tile 尺寸无关 |
| 1 ms | ❌ 未达成 |
| 2.3 ms 基线 | ✅ 保持（prefill 2339 µs / API 2225 µs / decode 41.5 µs）|
| 代码 | ✅ 零改动，工作树干净 |

## 6. 下一步建议（需要不同的手段）

外部调参已经走到头。要定位 `step>128` 的真实原因，建议：

1. **在 kernel 里加 `DumpTensor`/`printf`**，在 step=129 时打印标量侧所有与 step 相关的
   中间量（`mSizeAligned`、`dstStride`、各 offset），对比 128 与 129 的差异；
2. 或用 **`msprof op simulator`**（有指令级时间线）看崩溃前最后几条指令；
3. 若确认是标量侧 `uint16` 回绕，改动会很小（把中间量提升为 32 位）。

这三条都需要在 kernel 内部插桩，比外部参数试验更可靠，但也更耗时。
