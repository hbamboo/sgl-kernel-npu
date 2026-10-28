# 第四轮：7 号卡上的向量段重写尝试与结论修正（2026-10-28）

> 承接 `REPORT-round3-findings-20261028.md`。按用户要求在 **device 7** 上推进。
> 代码零改动（实验旋钮已完整回退，工作树与 `c267c2c` 一致）。
> **1 ms 仍未达成**，但本轮推翻了上一轮关于阻塞点的两个判断。

## 0. 守护点（device 7 实测）

| 项 | 值 |
|---|---|
| prefill T=30456（kernel） | **2347 µs**（224.5 TFLOP/s）|
| prefill T=30456（Python API + 真实权重）| **2230 µs** |
| decode T=1 / 47 层 | **40.9 µs / 1157 µs** |
| 工作树 | 干净，等同 `c267c2c` |

## 1. 纠正上一轮判断一：split-M 的 AIV 侧不是"每核 ≤2 token"

上一轮我写"限制是向量段每核 ≤2 token（`ceil(step/64)`）"。读 split-M 源码后确认这**不适用于 split-M**：

```cpp
// MlaPrologV3SplitM::AivProcess
curStepVecBackToken_ = curMSize / cvRatio_;    // cvRatio_ == 2
aivOffset.curVecToken = curStepVecBackToken_ + isBackVec * curStepVecFrontListNum_;
```

split-M 把每个 step 的 `curMSize` 在 **2 个 AIV 核**之间对分，所以 step=128 时每个 AIV 核处理
**64 个 token**，且 `RmsNormCq` / `RmsNormRopeScatterCkvKr` 内部本来就是逐 token 循环。
`ceil(step/64) ≤ 2` 那条是 **split-N**（`MlaPrologVecS1CubS2`）的约束，不是 split-M 的。

⇒ 上一轮"重写向量段让它能循环 >2 个 token"的施工方向**指向了错误的代码位置**。

## 2. 纠正上一轮判断二：真正的 step 上限是 L1_A，不是 L0A

本轮把 `baseK` / `baseN` / `stepK` 三个旋钮都接上（含上一轮漏掉的 mmQn `baseN`），
并用 dump 确认参数确实传进 kernel（`mxfp8MmBaseK=128 mxfp8MmBaseN=64 step=256`），然后实测。

推导出的容量模型：

```
L1_A:  A in L1 = stepBatchSize × baseK × stepK ≤ 128 KB
L0A :  align(step,16) × baseK ≤ 32 KB                    (fp8 权重)
L0C :  align(step,16) × baseN × 4 ≤ 64 KB
```

**upstream 默认值正好顶死 L1_A**：

| step | baseK | stepK | kL1StepSize | A in L1 | L1_A=128KB |
|---|---|---|---|---|---|
| **128** | **256** | **4** | **1024** | **131072** | **正好等于上限** ← 现状 |
| 256 | 256 | 4 | 1024 | 262144 | 超 2× |
| 256 | 128 | 4 | 512 | 131072 | 正好 |
| 192 | 128 | 4 | 512 | 98304 | 可 |

这也解释了为什么 `baseK=256` 时 step 只能是 128：**不是 L0A，是 L1_A 恰好被撑满**。

## 3. 但按模型调参后仍然越界 —— 说明还有第四个约束

按上表把所有深度降到"模型允许"的档位后实测：

| step | baseK | baseN | A in L1 | A in L0A | 结果 |
|---|---|---|---|---|---|
| 128 | 256 | 128 | 131072（满）| 32768（满）| OK（684 µs）|
| 128 | 128 | 64 | 65536 | 16384 | OK，但**变慢到 1137 µs** |
| 160 | 128 | 64 | 81920 | 20480 | **L0A 越界** |
| 192 | 128 | 64 | 98304 | 24576 | **L0A 越界** |
| 256 | 128 | 64 | 131072 | 32768 | **L0A 越界** |
| 256 | 64 | 32 | 65536 | 16384 | **L0A 越界** |

`step=160` 时 A tile 只有 20 KB（远低于 32 KB）、L1 只有 80 KB（远低于 128 KB），
**仍然报 "CUBE to operate L0A is out of bounds"，且 32 个核全部报错**。

⇒ 存在一个**不随 baseK/baseN 缩放的固定 L0A 占用**。结合 `mmLocalTensors<T,O_L0C>::Init`
用编译期常量 `L0A_PP_SIZE` 计算各 tensor 的 L0 偏移这一事实，可以判断：只要任何一个 GEMM 的
tile 超过 32 KB，后续 tensor 的偏移就会整体越界；而**这些偏移是按固定 `L0A_PP_SIZE` 硬算的**，
不是按实际 tile 大小。所以"压缩 baseK 换取更大 step"这条路在当前 kernel 的 L0 寻址方式下不成立。

另外 `baseK=128` 即使 step 不变也从 684 µs 退化到 1137 µs（cube 占比 34%→19%，
MTE2 占比 →83%）：减小 baseK 会让 L1→L0 的搬运轮次变多，本身就有代价。

## 4. 结论

| 结论 | 依据 |
|---|---|
| step 卡在 128 的**首要**原因 | L1_A：`step × baseK × stepK = 128 KB` 被 upstream 默认值正好撑满 |
| step > 128 的**真正**障碍 | L0 张量偏移按编译期 `L0A_PP_SIZE` 硬算，任一 GEMM 超 32 KB 即整体越界；实测 step=160（20 KB）也崩 |
| `ceil(step/64) ≤ 2` | 只对 **split-N** 成立，split-M 每 AIV 核处理 `step/2` 个 token |
| 减小 baseK 的代价 | 684 → 1137 µs（L1→L0 轮次增加），本身不划算 |

### 到 1 ms 需要什么（修正后）

passes 从 8 降到 2 仍是唯一杠杆（288 µs/pass × 8 = 2304 µs → 2 pass ≈ 576 µs）。
但**不能再靠调 baseK**，而必须：

1. **让 L0 张量偏移按实际 tile 大小计算**（改 `mmLocalTensors::Init` 的偏移逻辑），
   使 step 变大时各 buffer 能共存；或
2. **在核内对 step 做二层循环**：外层步长保持 128（L1/L0 都不变），内层把权重加载提到循环外，
   只让 A/B 的 *使用* 重复——这是真正能消除"每步重载权重"的做法，改的是
   `Process()` 的 `for (mOffset ...)` 与 `AicProcess` 的权重加载位置，不是向量段。

方式 2 与我上一轮给的方向不同（上一轮指向向量段），本轮把它修正为 **cube 侧的循环结构**。

## 5. 风险提示

本轮在 device 7 上触发了多次 507015（设备级异常）。device 7 上有常驻 `sglangscheduler`
（HBM 91%、84 GB），Aicore 空闲所以计算可用，但**每次异常都会打断同卡进程**。
若要继续，建议先确认该 sglang 进程是否可停，或另寻真正独占的卡。
