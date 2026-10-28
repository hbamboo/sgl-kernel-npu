# 推进 1 ms：step 循环重写改动清单

> 前置：`BASELINE-SNAPSHOT-20261028.md`（基线 `3ae0be3`，prefill 2336 µs / API 2230 µs）。
> 本文件是**动代码之前**的施工清单，需确认后再实施。

## 0. 状态：本计划已被 REPORT-1ms-accounting-20261028.md 修正

核算后发现"权重常驻 L1"不可实现（权重 1.1–3.8 MB vs L1_B 256 KB），且
"passes = mSubSize/step" 的真正机制是**每 step 重读权重**（B 占搬运量 97.8%），
不是 A 侧的读放大。修正后的结论与更聚焦的下一步见该文档。
本文件保留作为原始计划记录。

## 0b. 为什么是这个改动

已实测确定（第 5 轮）：`stepBatchSize ≤ 128` 是全路径硬边界，改 baseK/baseN/stepK 都解不开。
而 split-M 的时间与 pass 数线性：

```
t ≈ passes × 288 µs   +   token × 0.071 µs
passes = ceil(mSubSize / stepBatchSize)
```

| step | passes（T=30456, mSubSize=952）| 时间 |
|---|---|---|
| 128（现状）| 8 | **2336 µs** |
| 256 | 4 | ≈1.5 ms |
| 512 | 2 | ≈1.0 ms |

`passes` 之所以等于 `mSubSize/step`，是因为**每个 step 都要把整套权重重新灌进 L1/L0**：

```cpp
// kernel_mla_prolog_split_m.h:742
for (int64_t mOffset = mOffsetStart_; mOffset < maxMOffset; mOffset += baseParams_->stepBatchSize) {
    UpdateStepBatchParams(curMSize);
    ...
    AicProcess<>(aicOffset, mOffset, mmQnLoops);   // <-- 内部重新加载全部权重
}
```

`AicProcess` 每次进入都会把 `weightDqOffset/weightDkvKrOffset/weightUqQrOffset/weightUkOffset` 置 0
并把权重从 GM 搬到 L1（`MatmulSplitMKOuter` / `MatmulQcQr` / `MatmulQn*`）。
**所以真正要做的不是"让 step 变大"，而是"让权重只加载一次、在多个 step 之间复用"。**

## 1. 方案：权重常驻 + step 内层循环

把现在的单层循环改成两层，权重加载移到外层：

```
for (mOuter = mOffsetStart_; mOuter < maxMOffset; mOuter += WEIGHT_TILE_TOKENS) {
    加载权重到 L1（一次）                      <-- 现在每 step 都做
    for (mInner = mOuter; mInner < ...; mInner += stepBatchSize) {
        AicProcess(...)   // 只用已驻留的权重，不再搬
        AivProcess(...)
    }
}
```

`WEIGHT_TILE_TOKENS` = 一次权重驻留能服务的 token 数。理论上限由 B 侧（权重）的 L1 容量决定：
权重 tile 常驻后，A 侧每次仍需 `step × kL1StepSize`。若 A 侧继续按 128 token 走，
则 `WEIGHT_TILE_TOKENS` 可以是 952（整段），权重只搬 1 次。

## 2. 涉及改动点（文件 + 行号）

| # | 位置 | 改动 | 风险 |
|---|---|---|---|
| 1 | `kernel_mla_prolog_split_m.h:742` `Process()` | 双层循环，权重加载提外层 | **高**（barrier 结构）|
| 2 | `AicProcess()`（同文件 ~815） | 拆出"只加载权重"与"用权重计算"两段 | 高 |
| 3 | `MatmulSplitMKOuter`（`service_matmul_arch35.h:841`） | 支持"B 已在 L1，跳过 LoadL1B" | 中 |
| 4 | `MatmulQcQr` / `MatmulQnSyncDynamicQuantAndMulQr` | 同上，跳过 B 加载 | 中 |
| 5 | `CrossCoreSetFlag/WaitFlag` 序列 | 权重只加载一次后，flag 的"每 step 一轮"语义要重排 | **最高**（死锁风险）|
| 6 | `CalcWorkSpace`（`mla_prolog_tiling.cpp`） | workspace 按新结构重算 | 低 |
| 7 | 新增 tiling 开关 `weightResident` | 默认 0（走原路径），1 才启用新路径 | 低 |

## 3. 门控与回退

- 新路径全部藏在 `weightResident` 开关后，**默认 0**，旧路径代码不动；
- 旧路径必须保持逐位一致（用现有 A/B 脚本验证）；
- 任一步失败即 `git checkout` 回退（本轮及前几轮都验证过这个流程可行）。

## 4. 验收门（缺一不可）

| 门 | 标准 |
|---|---|
| 精度 | 与基线 `3ae0be3` **逐位相同**（五路 max\|d\| = 0），T=8192 / 30456 |
| 真权重 | 真实 checkpoint layer 2 五路 cosine ≥ 0.9999975 |
| 验收 | 27/27 |
| prefill | T=30456 < 2336 µs（目标 1.0–1.5 ms）|
| decode | T=1 ≤ 41.5 µs、47 层 ≤ 1150 µs（不得劣化）|
| 稳定性 | 连续 100 次调用无 507015；T 边界扫 1..30456 无挂死 |

## 5. 已知风险（必须先说明）

1. **barrier 重排是死锁高发区**。现在每个 step 有完整的 set/wait 配对；
   把权重加载提出去后，AIC 与 AIV 的 rendezvous 次数改变，配错就是 hang（我前几轮见过 300 s 超时）。
2. **无法像前面那样"参数级"回退**。这次改的是主计算路径，回退靠 git 而不是环境变量。
   所以我会：先提交基线 tag → 在分支上做 → 随时可 `git reset --hard`。
3. **设备异常**。前几轮已多次触发 507015；本轮在 device 0（干净、无进程）上做，
   但每次异常仍会中止当前进程。

## 6. 实施顺序（每步可单独验证）

1. 打 tag `baseline-2.3ms`，提交基线快照；
2. 只做"权重加载提外层"的**骨架**，先不保证正确，用 `DSH_MLA_V3_DUMP` 确认 pass 数下降；
3. 修精度到逐位一致（大概率是偏移/flag 问题）；
4. 补 tiling 开关与 workspace；
5. 全量验收 + 稳定性扫。
