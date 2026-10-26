# MlaPrologV3 for GLM-4.7-Flash —— 移植 / 适配 / 精度 / 性能 总结

> 代码位置：`csrc/mla_prolog_v3/`（算子本体）、`csrc/mla_prolog/`（其依赖的共享 tiling
> 与 arch35 kernel 头文件）。详细移植记录见同目录 `README-port.md`，每一处改动的脚本见
> `port_scripts/`。

## 1. 交付内容

| 项 | 内容 |
|---|---|
| 源码 | 从 `/mnt/paas/hxdnpu/ops-transformer/attention/mla_prolog_v3` 完整复制，另加其依赖的 `attention/mla_prolog`（op_host tiling + arch35 kernel） |
| 构建 | 纳入 sgl-kernel-npu 的 kernels 模块（A5 only），产出 `libsgl_kernel_npu.so`；**空目录全新构建已验证**（configure/build 均 rc=0、0 error、产物 3.08 MB） |
| 调用 | 三条路径：`torch.ops.npu.mla_prolog_v3`（分配输出）/ `torch.ops.npu.mla_prolog_v3_out`（调用方给缓冲区，解码快 13–14%）/ **PyBind 直调** `libsgl_kernel_npu.mla_prolog_v3_out`（跳过 dispatcher，再快 10%，T=1 = 32.9 µs）|
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

> 一次性前置：CANN 的 `merge_obj_text.sh` 会原地合并 device object，第二次增量构建会报
> `unknown file type`。先执行一次 `csrc/mla_prolog_v3/port_scripts/port_fix_merge_script.sh`
> （保留 `.orig` 备份、使合并幂等）；全新 build 目录则无需该补丁。

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
→ 41.6（核数启发式）→ 36.3（`_out` 变体）→ **32.9 µs（PyBind 直调）**；
旧轨迹续：
→ 41.6（核数启发式：T≤128 用 16 核）→ **36.3 µs**（`_out` 变体，调用方预分配缓冲，逐位一致）；
预填 T=256：118.8 → **90.0 µs**（mm1 最小瓦片 64→32 + D=192 每步 128 token）；T=2048：687.7 → **650.9 µs**。
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
3. **SGLang 接入（已提供免改代码的 shim）**：SGLang 的 `forward_mlaprolog` 调的是
   `torch.ops.custom.npu_mla_prolog_v3`，而本镜像的 `custom_ops` 没有该算子。新增
   `python/sgl_kernel_npu/sgl_kernel_npu/mla_prolog_v3_compat.py`，在 `custom` 命名空间
   注册同名算子并转发到本移植实现，**签名与 SGLang 的 kwargs 完全一致**
   （token_x / weight_dq / weight_uq_qr / weight_uk / weight_dkv_kr / rmsnorm_gamma_cq /
   rmsnorm_gamma_ckv / rope_sin / rope_cos / kv_cache / kr_cache / cache_index /
   dequant_scale_w_uq_qr / rmsnorm_epsilon_cq / rmsnorm_epsilon_ckv / cache_mode /
   query_norm_flag / weight_quant_mode），返回顺序也一致
   （q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm）：

   ```python
   import sgl_kernel_npu
   from sgl_kernel_npu.mla_prolog_v3_compat import register_custom_alias
   register_custom_alias()   # 若 custom_ops 已提供则自动跳过，不覆盖官方实现
   ```

   已用 `tests/test_sglang_alias.py` 按 SGLang 的真实 kwargs 验证：解析成功，
   三路输出 `q_nope (T,20,512)` / `q_pe (T,20,64)` / `qr (T,768)` 与直接调用
   **逐位一致**。注意权重需为 NZ 格式，MXFP8 权重需先反量化到 bf16 并置
   `weight_quant_mode=0`。

   **另外还有一个必踩的坑**：`mla_preprocess.py` 用
   `any(re.fullmatch(r".*kv_b_proj", l) for l in quant_config.ignore)` 决定是否走
   `forward_mlaprolog`，而 modelslim 的 `ModelSlimConfig` 是从 quant 描述里的
   `"ignore"` 键读取该列表——本 checkpoint 的 `quant_model_description.json`
   **没有这个键**（它是扁平的 权重名→dtype 映射，`kv_b_proj.weight` 标为 FLOAT），
   于是 `ignore == []`，**即使装了 custom_ops 也永远不会走 MLAPO 分支**。
   为此 compat 模块另外提供：

   ```python
   from sgl_kernel_npu.mla_prolog_v3_compat import enable_mlaprolog
   enable_mlaprolog()   # patch ModelSlimConfig.__init__，补上 kv_b_proj 标记
   ```

   `tests/test_sglang_routing.py` 验证了三件事：标记逻辑在 4 种 ignore 输入下都满足
   SGLang 的谓词、`custom::npu_mla_prolog_v3` 注册成功、`enable_mlaprolog()` 在本镜像
   里确实打上了补丁（`ModelSlimConfig._mlaprolog_patched == True`）。

   **第三个坑：SGLang 把 `weight_quant_mode` 硬编码成 1**。GLM 的 MXFP8 权重若已按上文
   反量化成 bf16，再用 mode=1 传入会被 tiling 拒绝。shim 因此在转发前按实际 dtype 推断模式
   （浮点权重→0、fp8+e8m0→3、int8→保留调用方取值），并丢掉该模式用不到的 scale 参数；
   `tests/test_sglang_mode_infer.py` 验证了 4 种 dtype 的推断结果，以及"按 SGLang 原样
   kwargs（mode=1 + bf16 权重 + scale 张量）调用"与直接 mode=0 调用**逐位一致**。

   **第四个坑（最隐蔽）：`q_b_proj` 是 ND 格式**。`mlaprolog_preprocess_weight` 只对
   融合后的 `qkv_a_proj` 做 `npu_format_cast`（→NZ），`q_b_proj.weight` 与 `w_kc` 原样
   传入（MXFP8 scheme 的 `process_weights_after_loading` 明确走"只转置、不转 NZ"分支）。
   而本 kernel 的三个投影矩阵必须是 NZ（ND 会算出 cosine 0.005 的垃圾值）。shim 因此
   在转发前对 `weight_dq/weight_uq_qr/weight_dkv_kr` 做 ND→NZ 转换并缓存（权重加载后
   静态，`clear_format_cache()` 可清缓存）。`tests/test_sglang_nd_weights.py` 模拟真实
   情形（qkv_a 为 NZ、q_b 为 ND、`weight_quant_mode=1`），结果与"全 NZ + mode 0"
   参考调用**逐位一致**，且第二次调用命中缓存后仍一致。

   **第五个坑：MXFP8 权重**。kernel 原生 MXFP8 路径在 `Hcq=768` 下 scale 映射有误，唯一
   数值正确的配置是 bf16 + mode 0。shim 因此在首次调用时把 `float8_e4m3` 权重按 e8m0 尺度
   反量化成 bf16 并缓存（同时支持算子 ABI 的 `[in,out]` 与 checkpoint 的 `[out,in]` 两种
   scale 朝向），随后把调用切到 mode 0。`tests/test_sglang_mxfp8_auto.py` 用 **真实 layer-2
   的 fp8 权重 + e8m0 scale**（按 SGLang 的转置/切分方式组织）直接喂给 shim，五路输出
   对 CPU golden 的 cosine 为 0.9999976 / 0.9999979 / 0.9999979 / 0.9999979 / 0.9999982。

   **与 SGLang 权重预处理的约定已逐项对齐**（读 `mla_preprocess.py` 源码确认）：
   `qkv_a_proj.weight.transpose(0,1)` 后按 `q_lora_rank` 切出 q_a / kv_a（即 [in, out]）、
   `npu_format_cast` 转 NZ、`kv_b_proj.weight.unflatten(0,(-1,D+v)).split([D,v],1)` 取
   `w_kc = [N, D, Hckv]`、`cos_sin_cache[positions]` 后 `chunk(2)` 再 `repeat(1,2)`
   （即 cos/sin 前后半重复）——与本算子的 ABI 完全一致。

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
