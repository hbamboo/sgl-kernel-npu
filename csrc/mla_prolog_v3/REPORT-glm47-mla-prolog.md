# MlaPrologV3 for GLM-4.7-Flash —— 移植 / 适配 / 精度 / 性能 总结

> 代码位置：`csrc/mla_prolog_v3/`（算子本体）、`csrc/mla_prolog/`（其依赖的共享 tiling
> 与 arch35 kernel 头文件）。详细移植记录见同目录 `README-port.md`，每一处改动的脚本见
> `port_scripts/`。

## 1. 交付内容

| 项 | 内容 |
|---|---|
| 源码 | 从 `/mnt/paas/hxdnpu/ops-transformer/attention/mla_prolog_v3` 完整复制，另加其依赖的 `attention/mla_prolog`（op_host tiling + arch35 kernel） |
| 构建 | 纳入 sgl-kernel-npu 的 kernels 模块（A5 only），产出 `libsgl_kernel_npu.so` |
| 调用 | `torch.ops.npu.mla_prolog_v3(...)`（Direct kernel launch，不经 aclnn/GE） |
| 测试 | `tests/`：冒烟、CPU golden、真实权重、RoPE 标定、bench、profiling |

构建：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B build -S .
cmake --build build --target sgl_kernel_npu -j 16
```

## 2. 为什么需要适配（GLM-4.7-Flash vs 内置 MlaPrologV3）

容器里 CANN 9.1.0 内置的 `MlaPrologV3` 只接受 DeepSeek MLA 几何：
`Hcq ∈ {1536, 2048}`、`N ∈ {1,2,4,…,128}`、权重仅 bf16/int8、无 MXFP8。
GLM-4.7-Flash 是 `He=2048, Hcq=768, Hckv=512, D=192, Dr=64, N=20` + MXFP8
(`float8_e4m3` + 每 32 元素 `e8m0`) 权重，实测被内置算子拒绝：

```
Parameter Hcq of MlaPrologV3 has incorrect value 768. It should be 1536 or 2048.
Parameter N   of MlaPrologV3 has incorrect value 20. It should be 1,2,4,...,128.
The dtype or format ... inconsistent with that defined in the operator prototype   (fp8 权重)
```

ops-transformer 的**源码**版本已经支持 `D=192`、`N ∈ [1,128]`、`weightQuantMode=3`(MXFP8)，
只有 `Hcq` 一道门还挡着 768，因此移植时只放开这一处即可。

## 3. 改了哪些地方（要点）

1. **放开 `Hcq=768`**（`mla_prolog_tiling_check.cpp`，唯一一处语义改动）。
2. **单入口 kernel + 按 tilingKey 运行时派发**：替代 `ASCENDC_TPL` 的“每组合一个
   kernel 二进制”机制（与仓库内 `csrc/compressor` 同思路）。
3. **补齐 direct-launch 环境缺失的上下文**：
   * `GET_TILING_DATA_WITH_STRUCT` 不存在 → 生成 GM→local 的逐字段拷贝；
   * `ASCENDC_TPL` 声明头在 host 编译遍不通过 → preamble 跳过；
   * **PA cache 的 `kvCacheStride0/krCacheStride0`** 原应由 GE descriptor 提供，direct
     launch 没有 → 初版漏填，导致 **T > blockSize 时 kv/kr cache 全部写错**（已修复）。
4. **平台信息与编译工具链**：用 `PlatformAscendCManager` 取代 `fe::PlatFormInfos*`；
   去掉 cxx11 ABI 不兼容的 `ge::TypeUtils::DataTypeToSerialString`；
   把 CANN 的 `merge_obj_text.sh` 改成幂等（否则第二次构建报 `unknown file type`）。
5. **MXFP8 支持**：`float8_e4m3fn`/`e8m0` 的 dtype 映射与 scale 张量形状对接。

## 4. 精度验证（目标要求“参考现有 CPU golden 并通过”）

自建 CPU golden（RMSNorm + 4 个 GEMM + interleave-half RoPE + PA scatter），
GLM 几何、T=8：

| 输出 | cosine | max abs diff |
|---|---|---|
| query (q_nope) | 0.999994 | 0.0030 |
| query_rope | 0.999996 | 0.0098 |
| query_norm | 0.999998 | 0.0173 |
| kv_cache | 0.999998 | 0.0169 |
| kr_cache | 0.999998 | 0.0007 |

覆盖 token 范围（修掉 PA stride 缺陷后）：

| T | query | query_rope | kv_cache | kr_cache |
|---|---|---|---|---|
| 8 | 0.999994 | 0.999996 | 0.999998 | 0.999998 |
| 128 | 0.999961 | 0.999994 | 1.000000 | 0.999999 |
| 256 | 0.999980 | 0.999982 | 0.999995 | 0.999999 |
| 512 | 1.000094 | 0.999975 | 0.999993 | 1.000000 |

**真实 checkpoint**（`/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8` 的
`model.layers.2`，MXFP8 权重按 e8m0 反量化到 bf16）：五路输出 cosine ≥ 0.9999975。

**与厂商二进制交叉验证**（内置算子支持的 DeepSeek 几何）：五路输出
**逐位一致**（max abs diff = 0.00000）。

两个必须知道的约定（用探针实测标定，写进 `README-port.md`）：

* `weight_uq_qr` 列是**按 head 交错**的：head h 占 `[h*(D+Dr), (h+1)*(D+Dr))`，
  前 D 列 `W_UQ`、后 Dr 列 `W_QR`；
* RoPE 是 **interleave-half**：`y[:Dr/2] = x[0::2]*cos[:Dr/2] - x[1::2]*sin[:Dr/2]`，
  `y[Dr/2:] = x[0::2]*sin[Dr/2:] + x[1::2]*cos[Dr/2:]`（cos/sin 前后半重复传入）；
* 三个投影权重必须是 **FRACTAL_NZ**（`torch_npu.npu_format_cast(w, 29)`）。

## 5. 性能（Ascend950DT）

机器上限实测：HBM **3.27 TB/s**，bf16 **424 TFLOP/s**（4096³ matmul），平衡点 130 FLOP/byte。

| T | 1 | 8 | 64 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| 总耗时 (µs) | 41.6 | 41.6 | 53.9 | 90.0 | 169.4 | 329.3 | 650.9 |
| kernel (µs) | 18.2 | 18.2 | 30.5 | 70.4 | 153 | 319 | 652 |
| roofline 占比 | 30% | 30% | 24% | 15% | 14% | 13% | 13% |

主机侧构成（已到框架下限）：torch dispatcher **16.7 µs** + kernel launch **5.3 µs** +
本算子逻辑 ~1.5 µs。任何 `torch.ops` 算子都要付这 16.7 µs。

优化轨迹（T=1）：77.7 → 48.5（tiling/workspace 缓存）→ 45.1（无分配哈希 key）
→ 41.6 µs（核数启发式：T≤128 用 16 核）；预填 T=256：118.8 → **93.8 µs**（mm1 最小瓦片 64→32）。
当前实现比 CANN 内置算子**快 15–17%**（同几何、逐位一致）。

瓶颈归因（msprof `PipeUtilization`）：

| 场景 | 关键管道 | 结论 |
|---|---|---|
| T=1（16 核，18.9 µs） | MTE2 47% / MTE1 47% / MAC 27% / scalar 34% | 被 L1/L0 搬运与标量簿记限制 |
| T=256（32 核，94 µs） | **MTE1 49%** / MTE2 34% / **MAC 20%** | cube 80% 空转，是“喂料”问题不是算力问题 |

已排除的优化项（都有实测数据）：核数扫描（32/24/16/8/4）、
mm1 最小瓦片（已固化为 32，+2%）、mm3 baseK、mm1/mm2 baseK、acgraph（单算子 replay 反而 150 µs）、
在 3510 上打开 split-K/N（设备报错 507015，arch gate 是必要的）。

## 6. 已知限制 / 后续路线

1. **MXFP8 原生路径**（`weight_quant_mode=3`）：下投影(mm2)数值完全正确（cosine 0.999999），
   但 `weight_uq_qr`(mm3) 的 per-32 scale 在 `Hcq=768`(24 个 scale/行，非 32B 对齐) 下
   映射错误（16 核 cosine 0.981、32 核越界溢出）。当前生产路径是**反量化到 bf16**（无损、已验证）。
2. **性能离 roofline 仍有 3–4x**：三条证据（解码 MTE2/MTE1 各 47%；预填 MTE1 49%/MAC 20%；
   各种 baseK 无效）表明瓶颈是“阶段化小瓦片搬运”，需要改写数据流：
   * 解码态：权重常驻 L1、按 token 流式的持久 kernel；
   * 预填态：按 GLM 的 Hcq=768 重排 NZ 分块，减少 k 方向切分与阶段间同步。
3. **SGLang 接入**：需要在 `mla_preprocess.py` 的 `forward_mlaprolog` 把调用从
   `torch.ops.custom.npu_mla_prolog_v3` 改成 `torch.ops.npu.mla_prolog_v3`（本容器
   `custom_ops` 包没有该算子），并显式打开该分支（当前 checkpoint 的量化描述里没有
   `ignore=kv_b_proj` 白名单）。

## 7. 复现命令

```bash
# 精度（CPU golden，GLM 几何；可 T=1..512）
T=256 python3 csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py
# 真实 checkpoint
python3 csrc/mla_prolog_v3/tests/test_real_glm_weights.py
# 与内置算子逐位对比（DeepSeek 几何）
python3 csrc/mla_prolog_v3/tests/cmp_builtin_numeric.py
# 性能
python3 csrc/mla_prolog_v3/tests/bench_mla_prolog.py
# 主机/kernel 拆分
python3 csrc/mla_prolog_v3/tests/split_host_kernel.py
```
