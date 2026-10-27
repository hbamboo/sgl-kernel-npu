# Kernel 改写方案（把利用率推向 roofline）

> 前置阅读：`REPORT-glm47-mla-prolog.md` 第 7 节与 `PERF-COUNTEREVIDENCE.md`（现状与反证链）。
> 本文件只描述"如果要突破 13–16% / 29–30% 该怎么做"，每条都给出**涉及文件、改动点、验收门、回退条件**。
> 需要：一张可独占的卡、数小时连续窗口、以及明确授权（原因：改 kernel 数据流会产生
> 设备级故障风险，前 42 轮里有两次 `507015` 属于此类，虽只影响本进程，但会打断共享卡上的他人任务）。

---

## 现状约束回顾（决定了改写的形状）

| 约束 | 数值 | 影响 |
|---|---|---|
| L1 总容量 | 512 KB，A/B 各 128 KB × ping/pong | 权重窗口深度已被占满 |
| `kL1StepSize` | `baseK × stepK = 512` | 同时顶到 B（128 列）与 A（m=128）上限 |
| mm1/mm2 分核 | 24 核 × 32 列 / 9 核 × 64 列 | 每核列数不影响阶段耗时（实测 <1%）|
| 预填 mm1+mm2 | 70 µs kernel 中约 49 µs | 优化的唯一落点 |
| 解码 kernel | 18.2 µs，其中固定开销 ≈16.5 µs | 与权重字节几乎无关 |

由第一条可知：**在不改结构的前提下不可能再加深流水**。要么换执行单元（向量化 GEMV），
要么换切分维度（K 切分），要么减少阶段（融合）。

---

## 成本分解（本轮实测，决定各方案的优先级）

两个扫描（He 只动 mm1/mm2；N 只动 mm3/mm4）+ T 扫描（动每步 token 数）给出：

| 组成（T=256, N=20） | 实测 | 占比 |
|---|---|---|
| mm1+mm2（随 He 变化） | ≈ 49 µs | 55% |
| mm3+mm4（随 N 变化：N=4→20 只涨约 11 µs） | ≈ 11 µs | 12% |
| 其余（kernel 固定开销 + 向量阶段） | ≈ 29 µs | 33% |
| **合计** | **89.7 µs** | |

补充：按 T 扫描（64/256/1024/2048 → 53.9 / 89.7 / 329.3 / 649.9 µs）拟合，**每 128-token 步 ≈ 40 µs**、
单次 kernel 固定开销 ≈ 34 µs；即一个 128-token 步里 mm1+mm2 ≈ 24.5 µs、mm3+mm4 ≈ 15 µs。

**结论**：下投影（mm1+mm2）确实是最大单项，而固定的 29–34 µs 也不小。融合 mm1+mm2 若能省掉
"一个阶段"的固定开销（约 7–12 µs/步），就是 8–13% 的预填收益——收益量级取决于固定开销
是否按阶段摊分，只能实测确认。

## 方案 A：融合 mm1 + mm2（改动最小）

**做法**：加载期把 `W_DQ [He, Hcq]` 与 `W_DKV/KR [He, Hcq+Dr]` 沿 N 拼成 `[He, 1344]`，
kernel 里只发一次 `MatmulSplitN`，输出落到同一块 workspace 的相邻两段。

| 项 | 内容 |
|---|---|
| 涉及文件 | `op_host/mla_prolog_tiling.cpp`（新增 `mm1N` = Hcq+Hckv+Dr 的 fill 分支）、`op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h`（`AicProcess` 去掉第二次 `MatmulSplitN`）、`op_host/mla_prolog_v3_host.{h,cpp}`（新增 fused 权重入参或把拼接放在 host 侧缓存）|
| 新增 tiling 字段 | `fusedDown`（0/1）、`mm12SingleCoreN`、`mm12BlockNum` |
| 数值可行性 | **已验证**：`probe_fused_down.py` 在 m=1/8/64/128/256 上比较 `x·[W_dq|W_dkv]` 与两次独立 GEMM，结果**逐位相同**（max\|d\| = 0.00e+00）——融合不引入任何数值差异 |
| 收益预期 | **8–13% 预填**（按上面的成本分解：每 128-token 步的 mm1+mm2 ≈24.5 µs，融合若能省掉一个阶段的固定开销即为 7–12 µs/步）。这个区间仍是估计，最终以改完后同场 A/B 实测为准；早先的"3–6%"与"50%"两个说法都已作废（前者基于 mm2 扩核实验的外推，后者是 `torch.matmul` 调用开销造成的假象）|
| 验收门 | 现有 22 项全过（尤其 T=8/256 golden 与厂商逐位）；预填 T=256 需要比 89.7 µs 有可复现下降（同场 A/B，≥2%）|
| 回退条件 | 任一项验收失败或无可复现下降即回退 |
| 风险 | 低（无新同步，且数值已证明逐位等价），但要保证向量侧读取偏移对两段都正确 |

---

## 方案 B：解码态持久 kernel（收益最大，解码 18.2 → 8~10 µs）

**为什么可行**：解码时 m=T≤16，cube 的 GEMV 效率只有 ~29%，而权重必须整块流过 L1。
把 m 方向交给向量单元（AIV）做 FMA、把权重用 MTE2 顺序流式送入 UB，就能把"每 128 列一次
L1→L0 事务"换成"一次大块 DMA + 向量乘加"，绕开 L1 双缓冲的容量墙。

**结构**（新 kernel，不动现有路径）：

```
mla_prolog_decode<<<32 AIC + 64 AIV>>>(x, W_dq, W_uqqr, W_uk, W_dkvkr, ...)
  phase 0  AIV: 量化/搬运 x（T×He，≤16 行）到 UB，常驻
  phase 1  AIV: c_Q  = Σ_k x[:,k] * W_dq[k, :]     每个 AIV 负责 Hcq/64 列，K 分块流式
           AIV: rmsnorm(c_Q) * gamma_cq
  phase 2  AIV: c_KV/k_R 同 phase 1（可与 phase 1 并行：不同 AIV 组）
           AIV: rmsnorm + rope + scatter 到 PA cache
  phase 3  AIC: q_c = c_Q · W_uq_qr（m=T，N=5120，可用 cube，32 核 × 160 列）
           AIC: q_n = q_c · W_uk（每头一个核）
  phase 4  AIV: rope(q_r) + 输出 query / query_rope / query_norm / dequant scale
```

| 项 | 内容 |
|---|---|
| 新增文件 | `op_kernel/mla_prolog_decode.cpp`（新入口）、`op_host/mla_prolog_decode_tiling.cpp` |
| 依赖 | 复用 `csrc/mla_prolog/op_kernel/arch35` 的向量原语（rmsnorm/rope/quant 都已有现成实现） |
| 收益预估 | 权重 17.3 MB 以 3.27 TB/s 需要 5.3 µs；向量 FMA 与 DMA 双缓冲后落在 8~10 µs，含固定开销后总时长 33 → 24~26 µs |
| 验收门 | 解码 T=1/8/16 的五路输出对 golden cosine ≥ 0.9999（当前 bf16 基线 0.999994）；T≥64 走旧路径，必须与其**逐位一致** |
| 回退条件 | 任何 golden 不达标、或解码未降 ≥20% 即整文件回退（新文件独立，风险可控）|
| 风险 | 中高：新 kernel 的同步（AIC/AIV 分工、PA scatter 的跨核序）需要仔细设计；建议先只支持 `kv_cache_quant_mode=0, weight_quant_mode=0` 一种组合 |

**为什么先把 m 方向交给 AIV**：cube 对 m=1 的利用率为 1/16（16×16 tile 只用一行），
而向量 FMA 按元素满负荷；权重流在两种方案里都是必须的，差别只在"进 L0 还是进 UB"。

---

## 方案 C：预填的 K 切分（预填下投影 −30%）

**做法**：让每个核拿到**完整 N**、只切 K（K/g 段），再用 workspace 做跨核归约。
这正是 kernel 里 `IsSplitKnSupported` 想做的事，但它被 arch gate 限定 DAV_9201，
且 `mla_prolog_kn_search.h` 对 N=768/576 产生不了候选。

| 项 | 内容 |
|---|---|
| 涉及文件 | `op_host/mla_prolog_kn_search.h`（放宽候选：允许 `ng=1, kg∈{2,4,8}`）、`op_host/mla_prolog_tiling.cpp`（放开 arch gate 前先确认 kernel 分支可用）、`op_kernel/arch35/*`（split-K 的 workspace 归约与 flag）|
| 验收门 | 预填 T=256/1024 golden 不变；T=256 预期 89.7 → 70~75 µs |
| 回退条件 | 出现 `507015`/结果错 → 立即回退并保留实验脚本 |
| 风险 | 高：kernel 侧该路径在 3510 上未验证过，可能出现设备级错误（前 42 轮已遇到两次）|

---

## 共同工程约定

1. **新路径一律走新 `tilingKey` 位**，旧路径不动，默认行为不变；
2. 每个方案都要在 `run_acceptance.sh` 里新增一项，且**旧 22 项必须全绿**；
3. 性能门限用同场 A/B（共享卡存在 ~33/48 µs 双稳态，必须交替测量）；
4. 每步完成后更新 `ACCEPTANCE-SNAPSHOT.md`（HEAD、.so md5、耗时、结果）；
5. 失败即回滚，并把实验脚本与结论一起归档到 `port_scripts/`（沿用现有惯例）。

## 建议顺序

A（半天，低风险，先拿 3–6%）→ B（数小时，收益最大，建议在独占卡上做）→ C（风险最高，视 B 的结果再决定）。
