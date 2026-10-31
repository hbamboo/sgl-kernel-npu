# FlashMLA in sgl_kernel_npu

This module migrates the Ascend 950 (arch35) FlashMLA operator into the standard
`sgl_kernel_npu` build and package. It retains the original general kernel and
96x112 tiling. Decode specializations and experimental performance changes are
excluded.

The operator supports BF16/FP16, BSND/BNSD/TND queries, TND-to-NTD output,
PA_BBND/PA_BNBD/PA_NZ KV, mask modes 0/3/4, query lengths, sink logits, LSE and
padding between KV blocks. Absorbed Q/K has 576 columns and V has 512 columns.

```python
from sgl_kernel_npu import flash_mla, flash_mla_metadata

md = flash_mla_metadata(
    cache_seqlens, num_heads_q=20, num_heads_kv=1,
    max_seqlen_q=1, max_seqlen_kv=max_context_len, layout_q="BSND",
)
out, lse = flash_mla(
    q, k_cache, block_table=block_table, cache_seqlens=cache_seqlens,
    metadata=md, softmax_scale=model_scale,
    max_seqlen_q=1, max_seqlen_kv=max_context_len,
)
```

Pass the original model attention scale. For the validated GLM configuration
this is `0.0625`; absorbing Q/K into 576 columns does not redefine the scale.

## Build and dependencies

The standard A5 build produces `libsgl_kernel_npu_flash_mla.so` in the package's
`lib/` directory. Importing `sgl_kernel_npu` loads the companion library.

```bash
cmake --build <build-directory> --target sgl_kernel_npu_flash_mla -j8
```

Metadata generation uses the installed native `FlashMlaMetadata` custom OPP.
Configure its vendor and op-api paths through `ASCEND_CUSTOM_OPP_PATH` and
`LD_LIBRARY_PATH`. The migrated API does not import the `cann_ops_transformer`
Python package or compile kernels at import time. The custom OPP remains a
runtime dependency.

## NPUGraph and lifetime

For changing sequence lengths, generate metadata inside graph capture so the
native scheduler runs again on replay. Layers in the same captured step can
share the resulting handle. Length tensors must remain on the query device.
Omitting metadata asks `flash_mla` to generate it with the same native operator.

Immutable tiling tensors and their pinned sources are retained per device and
stream, including when their first copy is captured. Per-call scratch belongs
to the current stream and graph memory pool. The arch35 system workspace adds
16 MiB to user scratch. The Python API clears retained tiling before NPU
shutdown.

## Validation

Tests compare supported layouts, masks, sinks/LSE and block padding with an
independent FP32 reference. They also exercise FakeTensor shape inference,
multiple streams and NPUGraph replay with changing device lengths, zero-length
sequences, page tails and long contexts. Run on an available Ascend 950 device:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 -m pytest -q csrc/flash_mla/tests
```

Vendor headers retain their CANN Open Software License notices. Sources come
from `sgl-kernel-npu-flash-mla/flash_mla/op_kernel` and its transitive common
headers. General operator headers retain the original implementation; the
additional launch-data structure supports direct invocation.
