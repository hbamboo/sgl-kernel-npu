# MlaPrologV3 — sgl-kernel-npu port notes

Source: `/mnt/paas/hxdnpu/ops-transformer/attention/mla_prolog_v3` (copied verbatim,
plus its shared implementation directory `attention/mla_prolog`, which the op includes
through `../../mla_prolog/...`).

Layout:

```
csrc/mla_prolog/                 # shared tiling + arch35 kernels (dependency of V3)
csrc/mla_prolog_v3/              # the op itself (copied from ops-transformer)
    op_host/                     # tiling host glue + torch entry (new files)
    op_kernel/                   # kernel entry (adapted, see below)
    compat/                      # ops_err.h from ops-transformer/common/include/err
    port_scripts/                # every edit applied to the copied sources
    tests/                       # smoke tests (bf16 + MXFP8)
```

## Why the port was needed

The CANN 9.1.0 binary `MlaPrologV3` installed in the container only accepts the
DeepSeek MLA geometry (`Hcq ∈ {1536, 2048}`, `N ∈ {1,2,4,...,128}`, int8/bf16
weights, no MXFP8).  GLM-4.7-Flash uses `He=2048, Hcq=768, Hckv=512, D=192,
Dr=64, N=20` with MXFP8 (`float8_e4m3` + per-32 `e8m0` scales) weights, which that
binary rejects at tiling time.

The ops-transformer **source** of the op is newer: it already supports `D=192`,
`N ∈ [1,128]` and `weightQuantMode=3` (MXFP8).  Only the `Hcq` gate still rejects
768, so the port keeps the tiling/kernels and drops that single gate.

## What was changed (all under `port_scripts/`)

| # | change | file |
|---|--------|------|
| 1 | platform query via `PlatformAscendCManager` (direct-launch host has no `fe::PlatFormInfos*`); record tiling-key template args in the tiling data | `mla_prolog_tiling.cpp`, `mla_prolog_tiling_check.cpp`, `mla_prolog_tiling.h`, `mla_prolog_tiling_data.h` |
| 2 | drop `ge::TypeUtils::DataTypeToSerialString` (CANN ships it with the pre-cxx11 ABI only) | `mla_prolog_tiling_check.cpp` |
| 3 | accept `Hcq=768` (GLM-4.7-Flash `q_lora_rank`) | `mla_prolog_tiling_check.cpp` |
| 5 | single named kernel entry + runtime dispatch on the tiling key (replaces the `ASCENDC_TPL` binary selection, same trick as `csrc/compressor`) | `mla_prolog_v3.cpp` |
| 7/8 | entry argument list fix; drop `EnableGroupComputeOpt=1` combinations (they mix the E8M0 scale with the f32 rotary dequant helper and do not compile upstream either; GLM tiling selects `groupOpt=0`) | `mla_prolog_v3.cpp` |
| 9 | `GET_TILING_DATA_WITH_STRUCT` is not available in the direct-launch build → generate a field-by-field GM→local copy | `mla_prolog_v3_tiling_copy.h` (+ `.port_gen_tiling_copy.py`) |
| 11 | MXFP8 E8M0 scales carried as uint8 tensors map to `DT_FLOAT8_E8M0` | `mla_prolog_v3_torch.cpp` |
| env | `merge_obj_text.sh` is not idempotent (it links the device object in place), so repeated `cmake --build` runs fail with `ld.lld: unknown file type`; the installed script is patched to skip already-merged objects (backup: `merge_obj_text.sh.orig`) | `port_scripts/.port_fix_merge_script.sh` |

New files (not from upstream):

* `op_host/mla_prolog_v3_host.{h,cpp}` — builds `optiling::MlaPrologContext` from plain
  tensor descriptions and runs the unmodified `RunBigKernelTiling`.
* `op_host/mla_prolog_v3_torch.cpp` — torch schema/impl (`torch.ops.npu.mla_prolog_v3`),
  output allocation, tiling-data cache, direct kernel launch.
* `op_kernel/mla_prolog_v3_kernel_preamble.h` — skips the `ASCENDC_TPL` declarations
  during the host pass of the kernel compile.
* `op_host/mla_prolog_v3_tiling_test.cpp` — standalone tiling driver (not built).

## Build

```bash
cd /mnt/paas/hxdnpu/sgl-kernel-npu
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B build -S .
cmake --build build --target sgl_kernel_npu -j 16
```

The op is A5-only (`SGL_KERNEL_ENABLE_A5_ONLY_OPS`), i.e. `SOC_VERSION=Ascend950PR_9599`
or another `Ascend950*` value.

## Verify

```bash
# tiling only (host-only binary, no device needed):
g++ ... mla_prolog_v3_host.cpp mla_prolog_v3_tiling_test.cpp mla_prolog_tiling.cpp \
      mla_prolog_tiling_check.cpp -o mla_tiling_test
./mla_tiling_test 2048 768 512 192 64 20 8 2 128 3 0   # GLM-4.7 MXFP8 → tilingKey 3933665

# device smoke tests (any free device):
ASCEND_RT_VISIBLE_DEVICES=5 python3 csrc/mla_prolog_v3/tests/smoke_mla_prolog_v3.py
ASCEND_RT_VISIBLE_DEVICES=5 python3 csrc/mla_prolog_v3/tests/smoke_mla_prolog_v3_mxfp8.py
```

## Status

* [x] copied + builds inside sgl-kernel-npu (`libsgl_kernel_npu.so` with
      `torch.ops.npu.mla_prolog_v3`)
* [x] GLM-4.7-Flash geometry accepted by the tiling (bf16 and MXFP8)
* [x] bf16 path runs on Ascend950DT and writes `kv_cache` / `kr_cache`
* [x] **precision (bf16, first stages)**: `c_Q` and `c_KV` match a CPU reference
      with cosine 0.999998 / 0.999998 and max abs error 0.013 (bf16 rounding).
      NOTE: `weight_dq / weight_uq_qr / weight_dkv_kr` must be **FRACTAL_NZ**
      (`torch_npu.npu_format_cast(w, 29)`); passing ND silently produces garbage.
* [ ] full CPU golden (q_nope, q_rope, kr_cache, rope convention)
* [ ] MXFP8 numerics: with E8M0 scales the first GEMM (`c_Q`) matches
      (cosine 0.9987), but the `weight_uq_qr` dequant path explodes
      (max |q| ~1e36 = the 0x7F e8m0 byte pattern reinterpreted as float32).
      Open: expected scale layout / dtype for the mm3 dequant, and whether
      `smooth_scales_cq` is required.
* [ ] performance tuning (bandwidth / compute utilisation)
