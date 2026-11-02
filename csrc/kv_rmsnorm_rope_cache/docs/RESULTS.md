# Measured results (developer self-validation)

> Naming update (2026-11-09): the records below retain their original filenames and profiler symbols. Current source: `csrc/kv_rmsnorm_rope_cache/`; current API: `torch.ops.npu.kv_rmsnorm_rope_cache`; current companion: `libsgl_kernel_npu_kv_rmsnorm_rope.so`.

Environment: Ascend950DT, physical NPU 7 only (`ASCEND_RT_VISIBLE_DEVICES=7`,
logical device 0), CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post6.  Kernel
build SoC `Ascend950PR_9599` (DAV_3510).  Development CLI model: DeepSeek
(`deepseek-v4.1-flash[1m]` via the user-provided Claude Code configuration); no
Anthropic model is claimed.

Build artifact:
`<kernel-worktree>/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu_kv_rmsnorm_unified.so`
(built by `cmake --build build-kvunified --target sgl_kernel_npu_kv_rmsnorm_unified`).

## Correctness

| Suite | Result |
|---|---|
| Operator matrix (`test_kv_rmsnorm_unified_cache.py`, small) | **40 passed, 0 failed, 4 skipped** (large-only marker off) |
| Large old-op oracle (`-k large`) | **4 passed** (T=3787/p65, T=15148/p7194 x BF16, FP16) |
| Framework routing (`test_kv_rmsnorm_unified_cache_route.py`) | **10 passed, 0 failed** |

Oracle A: installed `torch_npu.npu_kv_rmsnorm_rope_cache` called with the real
512/64 slices of the same unified pool.  For unique slots the returned
`k_rope`/`c_kv` rows, the whole cache, and untouched rows are **bitwise equal**
(including T=15148, pages=7194).  Oracle B: independent CPU FP32 golden of the
source semantics.  Observed maxima over the matrix: BF16 `max_abs` 1.95e-3,
FP16 1.22e-4; typical mean absolute error < 1e-7.  Tolerances used: BF16
rtol/atol 2e-2, FP16 3e-3.

Covered: T = 0,1,2,31,127,128,129,257,3787,15148; BF16 + FP16; pages 2/9/65
(exhaustive untouched-row checks) and 7194/15148; shuffled, boundary
(0/127/128/last), negative, out-of-range and duplicate indices; qkv stride 1344;
nonzero layer storage offset (other layers bitwise untouched); packed vs strided
equality; padded dummy-0; repeated calls; same-stream producer/consumer without a
host sync; side stream + events; FakeTensor/Meta; validation rejects (dtype,
shape, alias, page gap, 5D cache).

Duplicate live slots and padding slot 0 are documented as undefined races
(upstream inherited).  Observed interleaving is at 32-byte DMA-block granularity;
every block of the dummy row is owned by one contributing padded token, and
outputs are still golden.

## Full-path event timing

Both arms: real `[pages,128,1,576]` unified pool, old arm gets its true 512/64
slices; cache initialized once outside the interval; every call includes input
prep, output allocation, launch and cache mutation.  20 warmup calls, 40-50
measured reps, A/B order alternated each rep; `torch.npu.Event` around the
complete call (milliseconds converted to microseconds); outputs kept alive
through each interval and dropped before the next.  Candidate is loaded by
absolute path.

| T | pages | input | old median / p95 (us) | new median / p95 (us) | net saved (us) | ratio |
|---:|---:|---|---:|---:|---:|---:|
| 3787 | 7194 | packed 576 | 1531.4 / 1559.8 | 71.3 / 82.0 | 1460.1 | 21.5x |
| 15148 | 7194 | packed 576 | 1573.2 / 1617.7 | 108.9 / 135.2 | 1464.3 | 14.4x |
| 3787 | 128 | packed 576 | 188.8 / 220.3 | 73.8 / 83.6 | 115.0 | 2.6x |
| 3787 | 15148 | packed 576 | 3012.0 / 3028.9 | 72.1 / 83.8 | 2940.0 | 41.8x |
| 3787 | 7194 | strided 1344 | 1554.8 / 1572.6 | 71.5 / 88.1 | 1483.3 | 21.8x |
| 15148 | 15148 | packed 576 | 3047.5 / 3064.2 | 105.6 / 120.5 | 2941.9 | 28.9x |

A previous full run of the same final kernel gave 1524.8/67.5, 1558.1/100.0,
174.0/64.1, 3010.7/70.0, 1551.8/71.1 and 3046.6/101.1 us (old/new medians), i.e.
run-to-run variation below ~7%.  At fixed T the candidate is flat across cache
capacity (T=3787: 73.8 us at 128 pages, 71.3 us at 7194, 72.1 us at 15148;
T=15148: 108.9 us at 7194, 105.6 us at 15148), while the old path scales
linearly (~0.2 ms per 1000 pages).  Raw samples:
`results/bench_full_path.json`.

The event interval includes host launch latency when the device is idle (host
enqueue measured median ~46 us for the candidate, ~124 us for the old op).  In a
back-to-back burst the candidate costs ~45.5 us/call; separate kernel-only
profiler time is 21 us (T=3787) and 54 us (T=15148).

## Profiling (kernel sequence)

`results/profiles/` and remote `/mnt/.../results/profiles/`.  Cache 7194 pages
(1,060,798,464 bytes), 3 calls per window after warmup.

Old path, per call (T=3787): `aclnnContiguous_Slice` 101.9 us (64-wide cache),
`aclnnContiguous_Slice` 566.3 us (512-wide cache), `KvRmsNormRopeCache` 15.5 us,
`aclnnViewCopy` 118.3 us, `aclnnViewCopy` 583.2 us.  Exactly **two whole-pool
Slice + two whole-pool ViewCopy** per call; the copy kernels are independent of
T for fixed capacity (T=15148: 100.1/565.5/48.1/114.4/583.6 us).

Candidate path, per call: only `sgl_kv_rmsnorm_unified_cache_0`, 20.4 us
(T=3787), 54.1 us (T=15148).  Class counts per 3-call window:
old `{slice:6, viewcopy:6, kv_rmsnorm_cache:3}`; new `{slice:0, viewcopy:0,
unified:3}`.  No whole-cache Slice/ViewCopy/contiguous kernel appears.

## Framework routing

Opt-in `SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE=1` in
`forward_mha_prepare_npu` (DeepSeek-yarn unified BF16/FP16 PA_BNSD path).
The test selects the new op through the real hunk, verifies cache updates, and
verifies fallback to the original operator for: env off, PA_NZ, non-unified
pool, wrong geometry, FP32, and `torch.compile` active.  Graph capture is
excluded until a graph profile proves zero copies.  No whole-model service or
TTFT/QPS was run; none is claimed.

## Limitations

See `csrc/kv_rmsnorm_rope_cache/README.md`: non-quantized BF16/FP16 only,
PA_BNSD ND unified packed-576 only, no PA_NZ/block-table/NZ, fixed 512/64,
eager-only framework route, duplicate-slot behavior undefined.
