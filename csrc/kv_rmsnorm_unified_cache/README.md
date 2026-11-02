# kv_rmsnorm_unified_cache (Ascend950 / DAV_3510)

Direct cache-layout migration of the upstream fused RMSNorm+RoPE latent-cache
operator.  It writes the framework's real unified 576-wide latent cache in
place, touching only the mapped token slots.  The legacy framework call
(`torch_npu.npu_kv_rmsnorm_rope_cache` with the 512/64 slices of that unified
pool) makes two whole-pool `Slice` kernels and two whole-pool `ViewCopy`
kernels per layer; this operator removes them.

## Provenance

* Upstream: `/mnt/sfs_turbo/hxd/pkgs/ops-transformer`, release branch `9.1.0`,
  commit `6f0d3ec620b338cab4f2d00322ff2e0d141419d2`, operator
  `posembedding/kv_rms_norm_rope_cache`.
* License: the copied upstream kernel files retain their Huawei copyright and
  CANN Open Software License Agreement Version 2.0 header.  The full license
  text is in `LICENSE-CANN-OSL-2.0`.  New host/CMake/test/README files are
  BSD-3-Clause, matching the target repository.
* Target base: `sgl-kernel-npu` `opt/codex-1031` at
  `79ab3b050e7a9109eea593ce4d4bf491f1e66df6`, branch
  `feat/kv-rmsnorm-unified-cache-20261102`.

### Mapping and intentional edits

| Upstream file | Here | Change |
|---|---|---|
| `op_kernel/arch35/kv_rms_norm_rope_cache_regbase_base.h` | `op_kernel/kv_rms_norm_rope_cache_regbase_base.h` | Trimmed copy: only the non-quantized b16 RMSNorm helpers (`RmsNormVF`, `RmsNormBasicComputeVF`, `RmsNormPostVF`, `RmsNormPostWithMul`, dtype load/store, pad copy helpers) are retained. int8/hifloat8/fp8/fp4 quantization traits and `RmsNorm*Quant*` functions are deleted (out of scope). Arithmetic bodies are verbatim. `AscendC::Reg` is used instead of the device-only `AscendC::MicroAPI` alias so the same header also compiles in the target's `host_bisheng` precompile step (`MicroAPI` is `= Reg` on DAV_3510). |
| `op_kernel/arch35/kv_rms_norm_rope_cache_regbase_full_load.h` | `op_kernel/kv_rmsnorm_unified_cache.h` | Keeps `RopeVF`/`RopeBasicComputeVF` and the UB-tile data flow verbatim. Address changes only: kv source row pitch is `kvStride` (not 576), cos/sin pitch is `cosStride`, and the cache scatter addresses `slot*cacheDim` for latent and `slot*cacheDim+dv` for rope in one unified row. Slot ids are staged in UB once per tile; negative and out-of-range slots skip the cache write while still producing the returned rows. K/V scale/offset inputs, NZ/block-table cache modes, block-scatter paths and all quantization branches are removed. |
| `op_kernel/kv_rms_norm_rope_cache_apt.cpp` | `op_kernel/kv_rmsnorm_unified_cache.cpp` | Named entry `sgl_kv_rmsnorm_unified_cache`. Retains the upstream DAV_3510 SPR60 (`GLOBAL_OVERFLOW_MODE_CTRL`) save / set-0 / restore sequence around the computation. |
| `op_host/*` tiling | `op_host/kv_rmsnorm_unified_torch.cpp` + `op_kernel/kv_rmsnorm_unified_tiling.h` | The CANN `TilingContext` path is replaced by a POD tiling blob built directly from tensor metadata and platform UB size, following the target's `scatter_mla` / `qkv_rms_norm_rope_cache` direct-launch convention. |

No whole-cache scratch, mirror, packing or copy-back is introduced.  The kernel
reads only the current token rows of `kv`, and writes only `slot*576 .. +512`
and `slot*576+512 .. +64` for mapped slots.

## Interface

```python
torch.ops.npu.kv_rmsnorm_unified_cache(
    kv,            # [T,1,1,576], BF16/FP16, last dim contiguous,
                   # row stride may be >576 (e.g. the last 576 columns of [T,1344])
    gamma,         # [512], same dtype/device
    cos, sin,      # [T,1,1,64], same dtype/device; first/second 32-wide halves are
                   # used independently exactly as upstream
    index,         # int64 [T]; -1 (or <0) skips the cache write
    cache,         # [pages,128,1,576] or [rows,1,576], packed rows, data_ptr must
                   # already include the layer storage offset
    k_rope_out,    # [T,1,1,64] contiguous, caller-owned
    c_kv_out,      # [T,1,1,512] contiguous, caller-owned
    epsilon=1e-5,
) -> ()
```

`cache`, `k_rope_out` and `c_kv_out` carry mutable schema aliases
(`Tensor(a!)`, `Tensor(b!)`, `Tensor(c!)`) and a `Meta` implementation, so fake
tensor / meta flows work.  The Python-side wrapper for the framework gate lives
in the framework worktree:
`python/sglang/srt/hardware_backend/npu/modules/kv_rmsnorm_unified_cache_npu.py`.

### Supported (P0)

* BF16 and FP16, non-quantized; latent 512 / rope 64.
* `PA_BNSD` ND unified cache, page size 128, strictly packed
  `[pages,128,1,576]` per-layer view (or packed `[rows,1,576]`).
* `kv` row stride arbitrary but positive and >=576; no host `.contiguous()`.
* Empty T returns without launching.

### Explicitly rejected / fallback

* int8/hifloat8/fp8/fp4 cache quantization; K/V scale or offset inputs.
* `PA_NZ`, `PA_BLK_*` block-table modes.
* Any page gap or interleaved multi-layer page envelope (host rejects; the
  framework helper returns `None`).
* Head geometry other than 512/64, page size other than 128, dtypes other than
  BF16/FP16.
* Aliasing between the cache and any input, or between outputs and inputs/cache.
* The framework route is not taken under `torch.compile` or NPU graph capture
  (no graph-route copy profile yet; the original route stays).

### Index semantics

* Negative (`-1`) => skip the cache update, still return valid `k_rope`/`c_kv`
  rows (upstream behavior).
* `slot >= cacheRows` => skipped.  This is a deliberate hardening: upstream only
  checked `>= 0`.  It never adds a device->host read.
* Duplicate valid slots are **undefined**, as in the upstream multi-core
  scatter.  Latent and rope are written by two scatters, so a racing row can mix
  contributions; we do not promise last-writer-wins and do not detect it on the
  hot path.  Repeated padding writes to dummy slot 0 are the same race (observed
  at 32-byte DMA-block granularity); padding rows are not used for live-slot
  correctness.

## UB budget (why 512 B of RoPE workspace)

Installed CANN `kernel_utils_constants.h` defines `VECTOR_REG_WIDTH = 256` for
all architectures including DAV_3510, so upstream `platform::GetVRegSize()`
returns 256 B and `VL_FP32 = 64`.  `RopeBasicComputeVF` stages two full 64-lane
fp32 vectors, i.e. `2 * 64 * 4 = 512` B.  `KV_RMSNORM_VREG_BYTES` /
`KV_RMSNORM_VL_FP32` are shared by host and kernel in
`op_kernel/kv_rmsnorm_unified_tiling.h`; the kernel header
`static_assert`s them against `platform::GetVRegSize()`, and the host
`static_assert`s the same values, so the UB budget cannot silently diverge.
`ubFactor` is derived from the platform UB size with the upstream full-load
formula (double-buffered input/cos/out queues + RMSNorm workspace).

## Build and load (isolated, no install)

Inside `hxd-test`:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S <kernel-worktree> -B <kernel-worktree>/build-kvunified \
  -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
  -DBUILD_CATLASS_MODULE=OFF -DBUILD_TESTS=OFF
cmake --build <kernel-worktree>/build-kvunified \
  --target sgl_kernel_npu_kv_rmsnorm_unified -j16
# -> python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu_kv_rmsnorm_unified.so
```

Load by absolute path in a fresh process (no package install):

```python
torch.ops.load_library("/abs/path/libsgl_kernel_npu_kv_rmsnorm_unified.so")
```

The companion SO is also listed in the package loader
(`python/sgl_kernel_npu/sgl_kernel_npu/__init__.py`) for wheel builds, but the
tests and framework gate do not require it to be on `PYTHONPATH`.

## Tests

```bash
# dual oracle: installed old op on real unified 512/64 slices + independent CPU golden
ASCEND_RT_VISIBLE_DEVICES=7 KV_RMSNORM_UNIFIED_LIB=<abs .so> \
  python3 -m pytest -q csrc/kv_rmsnorm_unified_cache/tests/test_kv_rmsnorm_unified_cache.py
# large T=3787/15148 oracles
KV_RMSNORM_RUN_LARGE=1 ... -k large
# full-path event timing (see docs/RESULTS.md for measured numbers)
python3 csrc/kv_rmsnorm_unified_cache/tests/bench_full_path.py /tmp/bench.json
# kernel-sequence profile
python3 csrc/kv_rmsnorm_unified_cache/tests/profile_paths.py <outdir> 3787 7194
```

## Limitations

* No quantization, no PA_NZ, no block-table cache modes, no NZ layouts.
* Fixed 512/64 geometry; the coalesced scatter's `srcStride=0` is valid only
  because the host enforces `dkAlign==dk` and `dvAlign==dv`.
* Only packed per-layer 576 pools; a page-gap view is rejected rather than
  addressed with an extra page-stride parameter.
* Framework integration is opt-in and eager-only; whole-model TTFT/QPS were not
  measured, and no claim is made here.
