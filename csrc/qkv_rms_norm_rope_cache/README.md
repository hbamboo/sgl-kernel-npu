# qkv_rms_norm_rope_cache（sgl-kernel-npu 迁移版）

从 `ops-transformer/posembedding/qkv_rms_norm_rope_cache` 迁移到 `sgl-kernel-npu`，以 PyTorch
自定义算子形式注册为 `torch.ops.npu.qkv_rms_norm_rope_cache`。Ascend 950（DAV_3510）上用
RegBase（MicroAPI）实现，详见 [`OPTIMIZATION.md`](OPTIMIZATION.md)。

## 接口

```python
import sgl_kernel_npu  # 触发 libsgl_kernel_npu.so 加载，注册 torch.ops.npu.*
import torch

q_out, k_cache, v_cache = torch.ops.npu.qkv_rms_norm_rope_cache(
    qkv, q_gamma, k_gamma, cos, sin, index,
    q_out, k_cache, v_cache,
    num_head_q, num_head_k, num_head_v, epsilon)
```

| 参数 | 说明 |
| --- | --- |
| `qkv` | `[tokens, (Nq+Nk+Nv)*D]`，bfloat16 |
| `q_gamma` / `k_gamma` | `[D]`，bfloat16，RmsNorm 权重 |
| `cos` / `sin` | `[tokens, D]`，bfloat16（调用方传 `[T,1,1,D]` 时先 `.view(T, D)`） |
| `index` | `[tokens]`，int64，写入位置 = `block_id * block_size + offset`；`-1` 跳过 |
| `q_out` | `[tokens, Nq*D]`，bfloat16，原地输出 |
| `k_cache` / `v_cache` | `[block_num, N*D/16, block_size, 16]`，bfloat16（PA_NZ），原地 scatter |
| `num_head_q/k/v` | q/k/v 头数，要求 `Nk == Nv` |
| `epsilon` | RmsNorm eps |

计算语义与上游一致：SplitVD → q/k 的 RmsNorm（沿 D，权重 q_gamma/k_gamma）→ q/k 的
half-and-half RoPE（`y*cos + cat(-y[D/2:], y[:D/2])*sin`）→ q 写 `q_out`，k/v 按 `index`
scatter 到 PA_NZ 布局（等价于 `cache[blk, h, d1, off, d0] = value[token, h*D + d1*16 + d0]`）。

## 支持范围（相对上游的裁剪）

* dtype：仅 **bfloat16**（上游同时支持 float16）；
* cache 布局：仅 **PA_NZ**；
* k/v **不做量化**（不接收 scale/offset）；
* 不输出 `*_before_quant`；
* head dim 固定 **128**，`block_size` 需为 16 的倍数。

## 实现要点

* `op_kernel/qkv_rms_norm_rope_cache_regbase.h`：RegBase kernel。norm+RoPE 全程在向量寄存器内
  完成，每个 head 只有一次 UB 读 + 一次 UB 写；head_dim=128 正好两个整寄存器，无 mask/tail；
  一个 UB tile 只开一个 `__VEC_SCOPE__`，gamma 与每 token 的 cos/sin 只载入一次并复用。
* `op_host/qkv_rms_norm_rope_cache.cpp`：host launcher。三组核（Q / K / V）按实测每 head
  工作量分配（权重 Q=1.0 / K=1.7 / V=0.9）；tiling 为 POD 结构体，host 组好后拷到 device。
* kernel 源码从上游 `op_kernel/` 移植后按上面两点重写；`qkv_rms_norm_rope_cache_comm.h` 等
  UB-tile 旧实现已删除（从未被编译）。

## 与上游实现的差异

| 项 | 上游（CANN 算子工程） | 本迁移版 |
| --- | --- | --- |
| 入口 | aclnn / GEIR + `TilingContext` | `TORCH_LIBRARY_FRAGMENT(npu, m)`，`EXEC_KERNEL_CMD` 直调 |
| Tiling | `gert::TilingContext` + `SaveToBuffer` | 纯 POD 结构体，host 拷到 device |
| TilingKey | 按 cache_mode/quant 分发 | 单 kernel 变体，无 TilingKey |
| kernel | UB-tile（A3 风格） | RegBase（DAV_3510）+ 三组核分配 |
| workspace | `GetLibApiWorkSpaceSize()` | 无 |

## 构建

```bash
cd sgl-kernel-npu
./build.sh -a kernels Ascend950PR_9599     # A5；A2/A3 用 Ascend910B1 / Ascend910_9382
pip install --no-deps output/sgl_kernel_npu-*.whl
```

## 验证

```bash
pytest -q tests/python/sgl_kernel_npu/test_qkv_rms_norm_rope_cache.py
python benchmark/qkv_rms_norm_rope_cache/bench_vs_triton.py --tokens 34460 --warmup 20 --iters 50
```

* 精度：10 项用例全过（golden = 上游 ST 公式的 fp32 复算）；与 Triton 同输入对跑
  `q_out`/`k_cache` 0 offender（≤1 bf16 ULP），`v_cache` 逐 bit 相等（34460/32768/8192）。
* 性能（34460 token，q/k/v = 32/8/8，head_dim 128）：**597 us / 1448 GB/s**，
  Triton `split_qkv_rmsnorm_rope` 为 5 031 us，**快 8.4x**；32k 形状 567 us（8.4x）。

## 优化记录

所有优化项（原理 + 实测收益）见 [`OPTIMIZATION.md`](OPTIMIZATION.md)。精简版：

| 版本 | 时间（34460 token） | 有效带宽 |
| --- | ---: | ---: |
| 迁移初版（UB tile） | 5 152 us | 168 GB/s |
| RegBase 重写 | 2 250 us | 384 GB/s |
| 单 VEC_SCOPE + gamma/cos-sin 复用 | 1 047 us | 825 GB/s |
| 连续 slot 批量 scatter + ubRows=192 | 737 us | 1 173 GB/s |
| gamma 折进 RoPE 系数 | 699 us | 1 236 GB/s |
| 核数按实测工作量重分（39/16/9） | **597 us** | **1 448 GB/s** |

极限：Q 组每核约 1.1 M 条向量指令，实测 ≈1 条 VL 指令/cycle（fp32 64 lane），向量单元满吞吐；
与 memcpy roofline（3208 GB/s）的剩余差距来自算法固有的 fp32 向量指令数（≈20 条/head）。
