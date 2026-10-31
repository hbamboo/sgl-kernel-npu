# ScatterPaKvCache / MLA packed cache assembly

This port targets Ascend 950 (verified on physical Ascend950DT device 7, CANN 9.1, PyTorch/torch_npu 2.10). All original operator files, tiling branches and kernel classes are retained. The new C++ files adapt the registry context and launch typed AscendC entries using the package's current-stream OpCommand convention. Arithmetic compression retains numeric types; mixed K/V types in PA_NZ use independent byte widths. FP4 E2M1 x2 tensor metadata is converted to the source operator's logical last dimension.

## Build and call

From the sgl-kernel-npu repository root:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S . -B build -DSOC_VERSION=Ascend950DT_9582 -DBUILD_DEEPEP_MODULE=OFF
cmake --build build --target sgl_kernel_npu_scatter sgl_kernel_npu_scatter_mla -j8
export PYTHONPATH="$PWD/python/sgl_kernel_npu:$PYTHONPATH"
```

The normal package import loads the companion libraries alongside the existing library:

```python
from sgl_kernel_npu.scatter_pa_kv_cache import (
    scatter_pa_kv_cache, scatter_pa_kv_cache_mla,
)
scatter_pa_kv_cache(key, key_cache, slots, value, value_cache)
scatter_pa_kv_cache_mla(packed_kv, rotated_k_rope, slots, key_cache, value_cache, nope_dim=192)
```

Both APIs mutate caller-owned caches and return None. They also have torch.ops.npu schemas with cache mutation aliases and Meta dispatch. Warm each shape/attribute configuration before graph capture, as for sibling operators using TilingTensorCache. No global package installation is required for development.

`scatter_pa_kv_cache` follows the original Norm/PA_NZ and None/Nct/Rope/Omni/Alibi attributes; here Rope denotes source cache compression. Source kernel classes and all tiling formulas remain unmodified. In normal three-dimensional inputs, dimensions are tokens, heads, head size. Cache layout for Norm is pages, page size, heads, head size. Index tensors use int32 or int64. Each cache dtype matches its payload; PA_NZ permits different K and V dtypes. Inputs and caches must have a contiguous last axis for the source noncontiguous variants.

`scatter_pa_kv_cache_mla` takes contiguous BF16/FP16 `[T,H,nope+V]` plus already rotated `[T,1,R]`, writing Norm paged K and V directly. The first implementation requires positive head dimensions divisible by 16. Default tile size is four tokens; tuning experiments found 48-64 AIV cores near the measured copy ceiling. The internal torch.ops entry exposes tile/core/coalescing arguments for experiments. The normal Python API uses defaults.

The PTA entry exposes dtypes representable by this PyTorch build: BF16, FP16, FP32, signed/unsigned 8/16/32-bit integers, FP8 E4M3FN/E5M2 and packed FP4 E2M1 x2 on byte-copy paths. CANN-only HiFloat8 and FP4 E1M2 do not have corresponding PyTorch tensor dtypes in this environment; their original source declarations remain present. `ScatterPaCache` (single input/cache) is a separately registered source operator sharing this tiler, not this `ScatterPaKvCache` dual-input API.

## Validate

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 -m pytest -q \
  csrc/scatter_pa_kv_cache/tests/test_scatter_pa_kv_cache.py \
  csrc/scatter_pa_kv_cache/tests/test_scatter_compat.py \
  csrc/scatter_pa_kv_cache/tests/test_scatter_mla.py
```

SGLang integration is based on opt/p_kv_shared_tune. Enable `SGLANG_NPU_PREFILL_EXPANDED_KV=1` with the branch's existing PD prefill configuration, then `SGLANG_NPU_PREFILL_FUSED_KV_SCATTER=1`. Prefix batches carry packed tensors explicitly between prepare/core and write caches once; no-prefix batches keep contiguous FIA inputs and use the dual-cache scatter entry. Other layouts and piecewise compilation keep the original route. The latent cache sink and PD buffer layout remain intact. The switch defaults to off for controlled rollout.

## Measured evidence

Device7 standalone profiler, T=33305/H=20/nope192/rope64/V256, shuffled pages, ten adjacent ArgMax->target pairs: old layout plus two scatter kernels 2847.154us; fused 418.954us. Minimum logical read+write throughput is 3.063TB/s, compared with 3.302TB/s from a 1GiB source/destination copy on the same card (92.7%). These are effective bytes, not a physical HBM counter or proof of an absolute hardware maximum. Tile/core/coalescing scans and random slots reached the same plateau.

Real user weights `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`: BS12, 33305 new tokens and prefix128/request, ten alternating warmed A/B pairs gave complete ModelRunner extend median 786.044ms ->671.861ms (-14.53%). All logits were bitwise equal. Separate profiler: 94 generic scatters/102.808ms ->47 fused kernels/20.369ms. This is single-card model prefill, not HTTP TTFT or cross-card PD transport.


## Optional contiguous FIA outputs

The existing scatter_pa_kv_cache_mla entry accepts optional key_out[T,H,nope+rope]
and value_out[T,H,value_dim] keyword arguments. Supply both caller-owned,
contiguous outputs with the same dtype/device as packed_kv. They must not alias
the inputs, caches or each other. The kernel reuses assembled UB rows to emit
both dense FIA inputs and paged cache updates in one launch. Invalid PA slots
skip the cache update but still produce their dense rows. Existing calls without
these outputs keep the cache-only behavior. Warm each signature before capture.

The no-prefix SGLang path uses the dense outputs with its existing contiguous
FIA call. An older wheel is detected from the operator schema and retains the
previous K/V preparation path; upgrade the wheel to remove the five copy kernels.
