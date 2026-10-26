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
| 13 | tiling cache (shapes/dtypes/attrs) + reusable workspace: host cost 49 → 24 us/call |
| 15/18/23 | tuning knobs: `MLA_PROLOG_V3_CORES` (core count) and `MLA_PROLOG_V3_MAX_STEP` (per-step token cap) |
| 24 | **fix**: supply `kvCacheStride0` / `krCacheStride0`.  `ConvertContext` derives them from the GE tensor descriptor; the direct-launch host left both at 0, so the PA scatter ignored the block stride and every token spilling into block >= 1 (T > blockSize) landed in the wrong slot |
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

Verified across the token range after the PA-stride fix (CPU golden, GLM geometry):

Real checkpoint, per layer (five outputs, cosine):

| layer | 0 | 1 | 2 | 23 | 46 |
|---|---|---|---|---|---|
| worst output | 0.9999969 | 0.9999972 | 0.9999975 | 0.9999949 | 0.9999971 |

| T | query | query_rope | kv_cache | kr_cache |
|---|---|---|---|---|
| 8 | 0.999994 | 0.999996 | 0.999998 | 0.999998 |
| 128 | 0.999961 | 0.999994 | 1.000000 | 0.999999 |
| 256 | 0.999980 | 0.999982 | 0.999995 | 0.999999 |
| 512 | 1.000094 | 0.999975 | 0.999993 | 1.000000 |

Before the fix, T >= 192 produced garbage in `kv_cache` / `kr_cache` (all tokens
beyond the first PA block were written with stride 0).

## Performance (Ascend950DT) and roofline

Measured machine limits on this card:

* HBM bandwidth (256 MiB copy): **3.27 TB/s**
* bf16 matmul (4096^3): **424 TFLOP/s**
* machine balance: 424 / 3.27 = **130 FLOP/byte**
* **PyBind11 fast path**: `import libsgl_kernel_npu as _C; _C.mla_prolog_v3_out(...)`
  skips the torch dispatcher: T=1 36.7 -> **32.9 us** (-10.4%), T=8 35.4 -> 32.8 us,
  bit-identical results.  The binding lives in the same `libsgl_kernel_npu.so`
  (`mla_prolog_v3_pybind.cpp`), and `mla_prolog_v3_compat.direct()` returns it.
  The caller must pass bf16 FRACTAL_NZ weights - the dtype/format adaptation only
  happens on the torch.ops path.
* **host bookkeeping after the PyBind path** (measured with a wall-clock loop, T=1):
  empty binding call 0.11 us, full call 11.8 us, tiling-only (probe) 8.4 us, i.e.
  ~3.4 us launch + ~8.3 us in our own host code.  Most of that is rebuilding the
  `MlaPrologV3Tensors` description (20 `std::vector` allocations) and the tiling lookup.
  Trying to cache the description and revalidate it allocation-free was **slower**
  (15.3 us vs 11.8 us: the ~25 shape/dtype comparisons cost more than the rebuild), so
  that attempt was reverted - the simple path stays.
* **framework floor**: any NPU kernel op costs **~23.4 us** per call on this stack,
  broken down as **16.7 us pure torch dispatcher** (measured with a host-only op) +
  **~5.3 us kernel launch** (`aclrtLaunchKernelWithHostArgs` ≈ 3.9 us) + ~1 us glue.
  Our op adds ~1.5 us on top, so the host side is essentially at the framework floor.

Per call the op streams 17.3 MB of bf16 weights and does 17.3 MFLOP per token, so it
is memory bound up to T ~ 130 and compute bound above.

Latest timings (auto core count: 16 cores for T <= 128, 32 above):

| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 41.6 us (36.3 `_out`, **32.9 PyBind**) | 18.2 us | 5.3 us (BW) | 29% |
| 8 | 41.6 us (36.3 with `_out`) | 18.2 us | 5.4 us (BW) | 30% |
| 16 | 42.7 us | 19.3 us | 5.6 us (BW) | 29% |
| 64 | 53.9 us | 30.5 us | 7.4 us (BW) | 24% |
| 256 | 90.0 us | 66.6 us | 10.4 us (compute) | 16% |
| 512 | 169.4 us | 146.0 us | 20.8 us (compute) | 14% |
| 1024 | 329.3 us | 305.9 us | 41.8 us (compute) | 14% |
| 2048 | 649.9 us | 626.5 us | 83.6 us (compute) | 13% |

Note: this card occasionally flips into a second stable state (~48.5 us instead of
~41.5 us for decode, same binary/config), most likely other tenants on the shared
HBM/L2; all comparisons in this document use same-session A/B runs.

He scaling at T=256 (He=1024/2048/4096/8192 -> 71.9/93.8/143.3/244.5 us) shows the two
down-projection GEMMs (mm1+mm2) account for **~49 us of the ~70 us prefill kernel**
(70%), which is why they are the only meaningful prefill target.

Progression of the decode path: 77.7 us (first working version) -> 48.5 us (tiling +
workspace cache) -> 45.1 us (hash keys) -> 41.6 us (core-count heuristic) -> 36.2 us
(`_out`) -> **32.9 us** (PyBind), i.e.
**1.87x** faster than the first working port.

Core-count behaviour (`MLA_PROLOG_V3_CORES` overrides, default is the heuristic):

| cores | T=1 | T=8 | T=16 | T=32 | T=128 | T=256 | T=2048 |
|---|---|---|---|---|---|---|---|
| 32 | 44.6 | 44.5 | 45.4 | 47.5 | 59.4 | 95.8 | 688.7 |
| 16 | 42.1 | 42.6 | 42.1 | 45.1 | 58.2 | 111.7 | 858.5 |
| 8 | 44.4 | 48.3 | - | - | - | 137.1 | 1065 |

### Where the decode time actually goes

Sweeping the head count N (which scales the q_b / w_kc weight bytes) at T=1:

| N | weights | total | kernel | implied GB/s |
|---|---|---|---|---|
| 4 | 7.5 MB | 41.9 us | 18.5 us | 425 |
| 20 | 16.5 MB | 42.7 us | 19.3 us | 896 |
| 62 | 40.1 MB | 50.0 us | 26.6 us | 1580 |
| 124 | 75.0 MB | 59.6 us | 36.2 us | 2176 |

Fitting kernel time against bytes gives

```
t_kernel ≈ 16.5 us + 0.26 us/MB      (≈ 3.8 TB/s marginal, L2-assisted)
```

so for the real 16.5 MB of GLM prolog weights **~85% of the decode kernel time is
fixed overhead** (scalar bookkeeping, four serialised GEMM phase setups, cross-core
syncs) and only ~15-25% is weight streaming.  Consequences:

* MXFP8 weights (8.25 MB) would only save ~2 us at decode - not worth the numerics
  risk unless the fixed cost is removed first;
* the next real win has to come from the kernel structure (fewer phases / less
  per-phase scalar work), not from shrinking the weights.

aclgraph (`torch.npu.NPUGraph`) capture works and replays bit-exactly, but a
single-op graph is **slower** than eager on this stack: 150 us replay vs 36 us eager
with `mla_prolog_v3_out` (i.e. it is the graph-launch overhead, not the allocations
inside capture).  Graph capture is therefore only useful for the whole-model graph
SGLang builds, not per op; the `_out` entry point still matters there because it gives
stable buffer addresses.

Steady-state kernel profile, same script and device (msprof `PipeUtilization`, median):

| metric | before (32 cores, first port) | now (16 cores) |
|---|---|---|
| kernel time (T=1) | 28.96 us | **18.64 us (-36%)** |
| AIC MAC | 9.5% | **29.3%** |
| AIC MTE1 | 15.3% | 51.2% |
| AIC MTE2 | 35.9% | 48.9% |
| AIC scalar | 30.7% | 36.2% |

Prefill (T=256, 32 cores): MTE1 **56.4%** (busiest), MTE2 28.4%, MAC 17.9%, scalar 10% -
i.e. the L1->L0 feed, which matches the "L1 fully allocated" finding below.

Steady-state msprof `PipeUtilization` on the decode case (T=1, 16 cores, median task
duration **18.9 us**): AIC MTE2 **47%**, AIC MTE1 **47%**, MAC 27%, AIC scalar 34%,
fixpipe 6%; AIV scalar 30%, AIV vec 0.1%.  So the GEMV-shaped GEMMs are limited by the
L1/L0 feeding pipelines plus scalar bookkeeping, with roughly 2x headroom to the DMA
roofline - closing it needs deeper Matmul-API pipeline tuning (larger L1 stages) or a
purpose-built decode path, not a tiling knob.

Prefill (T=256, 32 cores, median task 94 us) shows the same picture, more strongly:
AIC MTE1 **49%**, MTE2 34%, **MAC only 20%**, scalar 13%.  So even at T=256 the kernel
is L1->L0 movement bound and the cube is starved - the "13% of the compute roofline"
is caused by feeding, not by the GEMM math.  Raising the mm3 baseK (64 -> 128, stepK
compensated) measured no change at all (93.8 / 93.9 / 93.9 us), so mm3's L0 tile shape
is not the limiter; the remaining gap needs a different data-flow (deeper L1 pipeline
or fusing the phases), i.e. kernel work.

**mm2 core-split does not help either** (measured): `FillMatmul2Tiling` hardcodes
`baseN = 64`, so mm2 runs 9 blocks x 64 columns while mm1 runs 24 x 32 - and since the two
phases are sequential, halving mm2's per-core columns looked like an easy win.  Making it
18 blocks x 32 columns (a legal 16-aligned tile) measured only 89.35 vs 89.75 us at
T=256 and 645.7 vs 649.8 us at T=2048 (<1%), so the phase is *not* limited by per-core
column count.  This also kills the "fuse mm1+mm2 for a bigger N per core" estimate: the
only benefit left there would be removing one phase boundary (~2-4 us of 70), i.e. 3-6%
of prefill - not worth a tiling + kernel-output-split rewrite.

**The kernel's L1 is 100% allocated** (`mla_prolog_comm_arch35.h`):

| constant | value | meaning |
|---|---|---|
| `L1_A_SIZE` | 128 KB, used as `* 2` (ping/pong) | A staging = 256 KB |
| `L1_B_SIZE` | 128 KB, used as `* 2` | B staging = 256 KB |
| total | **512 KB** | the whole L1 |

and the staging per iteration is `kL1StepSize = baseK * stepK = 128 * 4 = 512`, which is
*simultaneously* the B limit (128 cols x 512 x 2 B = 128 KB) and - at m = 128 - the A
limit (128 rows x 512 x 2 B = 128 KB).  So there is no headroom in either direction: a
larger baseK/stepK overflows (which is exactly the 507015 fault at baseK=256), and a
smaller baseK only makes the L0 tile shallower.  Decode does not help either: at m = 1
the A buffer is nearly idle but B still needs the full 128 KB, and `kL1StepSize` cannot
grow without overflowing B.

That is the quantitative reason the cube sits at ~7% of peak on these shapes: the kernel
streams weights through a saturated 512 KB L1 with 128 KB ping-pong buffers, so its
throughput is set by L1->L0 transaction granularity, not by anything tunable.

**Why the L0/L1 tile cannot simply be made larger** (probe, hardcoded values):

| mm1/mm2 baseK (bf16) | result |
|---|---|
| 128 (upstream) | works, 90.0 us @T=256 |
| 256, stepK=2 | faults (`507015`) once the step is 128 tokens (m=128) |
| 512, stepK=1 | faults immediately |

The kernel sizes its L1 staging as `128KB / (128 baseN * 4 stepK * sizeof(type))`, i.e.
it already sits at the L1 capacity limit; with m=128 the A tile adds another
`baseK*m*2` bytes, so 256/512 overflow.  (This also invalidates the earlier
"baseK 64/128 measured neutral" note: that probe went through an env knob that did not
actually reach the kernel - a hardcoded 256 faults, so it can never have been applied.)

Consequence for the remaining gap: with L1 full and 24 cores on mm1 (n=32 per core),
each core streams only ~131 KB per step yet the phase takes ~24 us, i.e. the cost is
per-transfer / scalar overhead, not bandwidth - and the tile cannot grow without
merging mm1+mm2 (n would become 56/core) or restructuring the phases.  Both are
kernel rewrites, not knobs.

Three experiments that did **not** pan out (reverted / kept opt-in):

* enabling the op's split-K/N schedule on DAV_3510 (`IsSplitKnSupported` is hard-gated
  to DAV_9201) faults the device (`507015`), so the arch gate is intentional;
* per-step tokens for D=192 are now **128** by default (upstream: 64).  Validated
  against the CPU golden at T=256/512/1024 and ~4% faster (T=256 93.8 -> 90.0 us,
  T=2048 675.6 -> 650.9 us); 192 tokens/step exceeds the UB budget and faults, so
  every other D keeps 64.
* `MLA_PROLOG_V3_MM1_MIN_TILE` changes how many cores the mm1 GEMM uses (64 -> 12
  cores for Hcq=768).  The port now defaults to **32** (24 cores at prefill): measured
  93.8 vs 96.0 us at T=256 and 675.6 vs 688.5 us at T=2048 (2 repeats each, stable to
  +-0.2 us), decode unchanged within noise.

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

## Remaining prefill headroom: why mm1/mm2 overlap is a kernel project

The He sweep shows the two down-projection GEMMs are ~70% of a T=256 kernel (49 of 70 us),
and they are independent: same input, different weights, different output buffers.  They
currently run **sequentially on the same block range** (`UsedBlockParams{0, mm1BlockNum}`
then `{0, mm2BlockNum}`), so only the phase boundary serialises them.

The kernel already knows how to run two sub-GEMMs on disjoint block ranges - the
group-compute path does exactly that for mm3 (`UsedBlockParams{0, QC_CORE_NUM}` and
`{QC_CORE_NUM, QC_CORE_NUM + QR_CORE_NUM}`, with offsets derived from
`blockIdx_ - QC_CORE_NUM` in `ComputeAicOffset`).  Mirroring it for mm1/mm2 is therefore
a small offset change plus a tiling choice (`mm1BlockNum + mm2BlockNum <= aicNum`, e.g.
16 + 16 so each GEMM serves a larger N per core).

What makes it a project rather than a patch is the cross-core flag protocol:
`CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ)` is phase-coupled to the
vector side, and `enableGroupComputeOpt` exists precisely to make a second, disjoint
phase legal - it is gated to int8 half quant with `G=8` (`GROUP_COMPUTE_*`) and its
template combination does not compile for our dtype on this toolchain (see the porting
notes).  A bf16 overlap therefore needs that protocol replicated (or a new flag scheme),
which is upstream-scale kernel work with deadlock risk on a shared card.

Measured follow-up (this was checked, not assumed): the overlap has **no headroom**.
At T=256 mm1 uses 24 of 32 cores (32 columns each) and mm2 uses 24 (24 columns each);
both are throughput-bound, so moving mm2 onto the 8 idle cores would need 576/8 = 72
columns per core - roughly 3x the per-core work - and end up slower than the 18 us it
takes sequentially.  The only real lever is split-K (full N per core, K sliced), whose
search (`mla_prolog_kn_search.h`) returns no candidate for GLM's shapes and whose kernel
path is gated to DAV_9201.  Bypassing that gate changes nothing in the tiling, which
confirms the schedule simply is not available for this geometry.

## Clean-build verification (done from an empty build dir)

```
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
      -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
      -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
      -DBUILD_KERNELS_MODULE=ON -B /tmp/bk_clean -S .
cmake --build /tmp/bk_clean --target sgl_kernel_npu -j 16      # rc=0, 0 errors
```

Verified: configure rc=0, build rc=0, 0 `error:` lines, `libsgl_kernel_npu.so` (3.08 MB)
produced, and all 12 acceptance checks pass against that fresh binary.

Reproduction drill (clean build dir + cleared ``__pycache__``): configure < 10 s,
**build 217 s** (3.6 min, -j 16), **acceptance 112 s** (12/12) - about 5.5 minutes from
scratch to green with no manual steps.

**Build prerequisite (once per CANN installation)**: CANN's
`.../tikcpp/ascendc_kernel_cmake/legacy_modules/util/merge_obj_text.sh` links device
objects in place, which breaks on the second build (`unknown file type`).  Run
`port_scripts/port_fix_merge_script.sh` once (it keeps a `.orig` backup and makes the
merge idempotent); with it applied, incremental rebuilds work.  A fresh build directory
always works even without the patch.

## Acceptance runner

```bash
bash csrc/mla_prolog_v3/tests/run_acceptance.sh 5    # device id, logs in /tmp/mla_prolog_v3_acceptance
```

Runs 11 checks in one go (CPU golden at T=8/T=256, real MXFP8 layer-2 weights, bit-exact
comparison against the vendor operator, the `_out` variant, the four SGLang shim tests,
the MXFP8 scheme-state reconstruction, and the performance sweep) and prints a summary.
It includes a token-boundary sweep (`sweep_tokens.py`): 31 sizes around every internal
tile boundary (1..1000) all stay >= 0.9995 on the five outputs.

It runs four real-checkpoint layers (0/2/23/46) so the validation is not tied to one layer.

Last run: **16/16 pass**, decode 36.2 us (`_out`) / 32.9 us (PyBind), T=256 89.8 us,
T=2048 649.9 us.

## Status

* [x] copied + builds inside sgl-kernel-npu (`torch.ops.npu.mla_prolog_v3`)
* [x] GLM-4.7-Flash geometry accepted (bf16 and MXFP8)
* [x] CPU golden + precision pass (bf16), synthetic and real checkpoint weights
* [x] host overhead reduced (tiling cache + workspace reuse)
* [x] `mla_prolog_v3_out` entry point (caller-provided output buffers): **bit-identical**
      to the allocating entry and **13-14% faster at decode** (T=1: 41.8 -> 36.3 us,
      T=8: 42.4 -> 36.3 us) because the five `at::empty` calls disappear; no change at
      T=256.  This is also the entry point the serving stack should use: one set of
      buffers per layer, stable addresses for ACL graph capture.
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
