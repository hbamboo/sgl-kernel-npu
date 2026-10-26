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

## Performance (Ascend950DT, practical peak BW 3.27 TB/s)

| T | latency | effective BW | GEMM throughput |
|---|---|---|---|
| 1 | 48.5 us | 357 GB/s (11% peak) | 0.36 TFLOP/s |
| 8 | 48.1 us | 364 GB/s (11%) | 2.9 TFLOP/s |
| 256 | 96.0 us | 256 GB/s (8%) | 46.1 TFLOP/s |
| 2048 | 688 us | 109 GB/s (3%) | 51.5 TFLOP/s |

Host cost 27 us/call (was 49). Decode is weight-bandwidth bound: 16.5 MiB bf16 at
3.27 TB/s implies a ~5 us floor; native MXFP8 weights (8.25 MiB) would halve it.

## Status

* [x] copied + builds inside sgl-kernel-npu (`torch.ops.npu.mla_prolog_v3`)
* [x] GLM-4.7-Flash geometry accepted (bf16 and MXFP8)
* [x] CPU golden + precision pass (bf16), synthetic and real checkpoint weights
* [x] host overhead reduced (tiling cache + workspace reuse)
* [ ] native MXFP8 numerics: first GEMM correct, `weight_uq_qr` dequant explodes for
      `Hcq=768` (24 scale bytes/row, not 32 B aligned) while `Hcq=1536` is fine
* [ ] kernel-side performance: ~21 us at T=1 vs a ~5 us bandwidth floor
