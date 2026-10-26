# MlaPrologV3 port notes (sgl-kernel-npu)

Source: `/mnt/paas/hxdnpu/ops-transformer/attention/mla_prolog_v3` (copied verbatim,
plus its shared implementation directory `attention/mla_prolog`, which the op includes
through `../../mla_prolog/...`).

```
csrc/mla_prolog/                 # shared tiling + arch35 kernels (dependency of V3)
csrc/mla_prolog_v3/              # the op itself (copied from ops-transformer)
    op_host/                     # tiling host glue + torch entry (new files)
    op_kernel/                   # kernel entry (adapted)
    compat/                      # ops_err.h from ops-transformer/common/include/err
    port_scripts/                # every edit applied to the copied sources
    tests/                       # smoke / golden / benchmark scripts
```

## Why the port was needed

The CANN 9.1.0 binary `MlaPrologV3` in the container only accepts the DeepSeek MLA
geometry (`Hcq ∈ {1536, 2048}`, int8/bf16 weights, no MXFP8).  GLM-4.7-Flash uses
`He=2048, Hcq=768, Hckv=512, D=192, Dr=64, N=20` with MXFP8 weights and is rejected at
tiling time.  The ops-transformer **source** is newer (supports `D=192`, `N ∈ [1,128]`,
`weightQuantMode=3`); only the `Hcq` gate rejects 768, so the port drops that gate.

## Changes (scripts under `port_scripts/`)

| # | change |
|---|--------|
| 1 | platform query via `PlatformAscendCManager`; record tiling-key template args in the tiling data |
| 2 | drop `ge::TypeUtils::DataTypeToSerialString` (pre-cxx11 ABI only in CANN) |
| 3 | accept `Hcq=768` (GLM-4.7-Flash `q_lora_rank`) |
| 5 | single named kernel entry + runtime dispatch on the tiling key (replaces `ASCENDC_TPL` selection, same trick as `csrc/compressor`) |
| 7/8 | entry argument list fix; drop `EnableGroupComputeOpt=1` combinations (do not compile upstream either) |
| 9 | `GET_TILING_DATA_WITH_STRUCT` is unavailable in the direct-launch build → generated field-by-field GM→local copy |
| 11 | MXFP8 E8M0 scales carried as uint8 map to `DT_FLOAT8_E8M0` |
| 12 | torch `float8_e4m3fn` → `DT_FLOAT8_E4M3FN` |
| 13 | tiling cache (shapes/dtypes/attrs) + reusable workspace: host cost 49 → 27 us/call |
| env | CANN `merge_obj_text.sh` made idempotent (backup `merge_obj_text.sh.orig`) |

New files: `op_host/mla_prolog_v3_host.{h,cpp}` (runs the unmodified
`RunBigKernelTiling`), `op_host/mla_prolog_v3_torch.cpp` (torch schema/impl +
launch), `op_kernel/mla_prolog_v3_kernel_preamble.h`, `op_host/mla_prolog_v3_tiling_test.cpp`.

## Build

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B build -S .
cmake --build build --target sgl_kernel_npu -j 16
```

## Interface notes

* `weight_dq`, `weight_uq_qr`, `weight_dkv_kr` must be **FRACTAL_NZ**
  (`torch_npu.npu_format_cast(w, 29)`); ND silently produces garbage.
* `weight_uq_qr` columns are **per-head interleaved**: head `h` owns
  `[h*(D+Dr), (h+1)*(D+Dr))`, first `D` = W_UQ, next `Dr` = W_QR.
* RoPE is **interleave-half**:
  `y[:Dr/2] = x[0::2]*cos[:Dr/2] - x[1::2]*sin[:Dr/2]`,
  `y[Dr/2:] = x[0::2]*sin[Dr/2:] + x[1::2]*cos[Dr/2:]`
  (callers pass cos/sin with duplicated halves, SGLang style).
* MXFP8 weights: either native (`weight_quant_mode=3`, not yet numerically correct
  for `Hcq=768`) or dequantised once to bf16 (`w * 2**(e-127)` per 32-block) with
  `weight_quant_mode=0` — exact, and what the validated tests use.

## Tests

```bash
python3 csrc/mla_prolog_v3/tests/test_glm_mla_prolog_precision.py  # CPU golden, 5 outputs
python3 csrc/mla_prolog_v3/tests/test_real_glm_weights.py          # real checkpoint layer 2
python3 csrc/mla_prolog_v3/tests/bench_mla_prolog.py               # latency / BW / TFLOP/s
python3 csrc/mla_prolog_v3/tests/split_host_kernel.py              # host vs device split
python3 csrc/mla_prolog_v3/tests/probe_rope_map.py                 # RoPE element mapping
```

## Precision (bf16, validated)

T=8, He=2048, Hcq=768, Hckv=512, D=192, Dr=64, N=20:

| output | cosine | max abs diff |
|---|---|---|
| query (q_nope) | 0.999994 | 0.0030 |
| query_rope | 0.999996 | 0.0098 |
| query_norm | 0.999998 | 0.0173 |
| kv_cache | 0.999998 | 0.0169 |
| kr_cache | 0.999998 | 0.0007 |

Real checkpoint (`model.layers.2`, MXFP8 weights dequantised to bf16): all five
outputs cosine >= 0.9999975, max abs diff <= 0.057.

## Performance (Ascend950DT) and roofline

Measured machine limits on this card:

* HBM bandwidth (256 MiB copy): **3.27 TB/s**
* bf16 matmul (4096^3): **424 TFLOP/s**
* machine balance: 424 / 3.27 = **130 FLOP/byte**
* **framework floor**: any NPU kernel op costs **23.4 us** per call on this stack
  (`helloworld`, tiny `qkv_rms_norm_rope_cache`).  The op itself adds ~0.7 us on top.

Per call the op streams 17.3 MB of bf16 weights and does 17.3 MFLOP per token, so it
is memory bound up to T ~ 130 and compute bound above.

Latest timings (auto core count: 16 cores for T <= 128, 32 above):

| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 41.6 us | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.7 us | 18.3 us | 5.4 us (BW) | 29% |
| 16 | 41.4 us | 18.0 us | 5.6 us (BW) | 31% |
| 64 | 54.3 us | 30.9 us | 7.4 us (BW) | 24% |
| 256 | 96.0 us | 72.6 us | 10.4 us (compute) | 14% |
| 2048 | 688.7 us | 665.3 us | 83.6 us (compute) | 13% |

Progression of the decode path: 77.7 us (first working version) -> 48.5 us (tiling +
workspace cache) -> 45.1 us (hash keys) -> **41.6 us** (core-count heuristic), i.e.
**1.87x** faster than the first working port.

Core-count behaviour (`MLA_PROLOG_V3_CORES` overrides, default is the heuristic):

| cores | T=1 | T=8 | T=16 | T=32 | T=128 | T=256 | T=2048 |
|---|---|---|---|---|---|---|---|
| 32 | 44.6 | 44.5 | 45.4 | 47.5 | 59.4 | 95.8 | 688.7 |
| 16 | 42.1 | 42.6 | 42.1 | 45.1 | 58.2 | 111.7 | 858.5 |
| 8 | 44.4 | 48.3 | - | - | - | 137.1 | 1065 |

msprof `PipeUtilization` on the decode case (T=1, 32 cores) shows where the remaining
time goes: MTE2 (weight DMA) 36%, **AIC scalar 31%**, MTE1 15%, MAC 9.5% - i.e. the
GEMV-shaped GEMMs are limited by scalar bookkeeping and small DMA requests, not by
the cube or by raw bandwidth.

## Cross-check against the CANN built-in op

On the DeepSeek-like geometry the built-in binary supports
(`He=7168, Hcq=1536, Hckv=512, D=128, Dr=64, N=16`, bf16), the ported kernel is
**bit-identical** to `torch_npu.npu_mla_prolog_v3` (max abs diff 0.00000 on all five
outputs, `cmp_builtin_numeric.py`) and **15-17% faster** end-to-end because it skips
the aclnn/op-plugin dispatch:

| T | ported | CANN built-in |
|---|---|---|
| 1 | 56.7 us | 66.6 us |
| 8 | 56.1 us | 67.9 us |
| 64 | 66.8 us | 78.4 us |

This also shows that the ~800 GB/s the kernel achieves is inherent to the op (not a
porting regression); improving it needs kernel work.

## Status

* [x] copied + builds inside sgl-kernel-npu (`torch.ops.npu.mla_prolog_v3`)
* [x] GLM-4.7-Flash geometry accepted (bf16 and MXFP8)
* [x] CPU golden + precision pass (bf16), synthetic and real checkpoint weights
* [x] host overhead reduced (tiling cache + workspace reuse)
* [ ] native MXFP8 numerics for `Hcq=768` (`test_mxfp8_precision.py`, T=8):

      | path | 16 cores | 32 cores |
      |---|---|---|
      | kv_cache / kr_cache (mm2) | cosine 0.9999989 | cosine 0.9999989 |
      | query / query_rope (mm3+mm4) | cosine 0.981 | garbage (1e37) |

      So the down-projection GEMMs are fine; the `weight_uq_qr` per-32 scale mapping is
      off by ~2% at 16 cores and reads out of bounds at 32 cores (core-split offset
      computation with 24 scale bytes per row).  bf16 (dequantised) weights stay the
      validated path.
* [x] performance baseline + roofline: host at framework floor (23.4 us), kernel at
      24% (decode) / 13% (prefill) of the measured roofline
* [x] decode core-count heuristic (16 cores for T <= 128): 45.1 -> 41.6 us
* [ ] kernel-side performance work: the profiler shows 31% AIC scalar + 36% MTE2 at
      decode, so the next step is a GLM-specific GEMV path (larger DMA requests, less
      per-token scalar work) and/or fp8 weight streaming to halve the bytes
