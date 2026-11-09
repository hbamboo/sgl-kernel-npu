# FlashMLA in sgl_kernel_npu

This module migrates the Ascend 950 (arch35) FlashMLA operator into the standard
`sgl_kernel_npu` build and package. It follows the upstream `ops-transformer`
single-entry structure: one kernel entry source, `op_kernel/flash_mla.cpp`, whose
template arguments are declared in
`op_kernel/arch35/flash_mla_template_tiling_key.h` with
`ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL`. The original 96x112 tiling and the
accepted direct-invocation algorithms are unchanged.

## Tree

```
csrc/flash_mla/
  CMakeLists.txt
  README.md
  LICENSE
  SOURCE_ORIGIN.json
  op_host/
    flash_mla_host.h          # shared metadata host declaration
    flash_mla_torch.cpp       # Torch schema, checks, tiling, template launch
  op_kernel/
    flash_mla.cpp             # the only kernel entry (upstream template shape)
    flash_mla_tiling_copy.h   # direct-invocation tiling POD copy
    flash_mla_shared.h        # metadata layout constants (also used by metadata)
    arch35/                   # FlashMLA attention algorithm headers
    utils/                    # FlashMLA type/config/copy helpers
    common/                   # common arch35/vf/memcopy header closure
```

There is no `vendor/` mirror and no per-dtype/per-layout entry wrapper.

## Kernel entry and template selection

The entry is:

```cpp
template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
__global__ __aicore__ void flash_mla(/* query, kCache, blockTable, ... , tiling */);
```

`arch35/flash_mla_template_tiling_key.h` is the selection source of truth:

- `InOutLayoutType`: BSND(0), BNSD(1), TND(2), TND_NTD(4);
- `KvLayoutType`: PA_BBND(1), PA_BNBD(2), PA_NZ(3);
- `HasAttenMask`: false/true;
- `Config`: 0 (`S1=64`, `S2=128`, `D=576`, `DV=512`), which maps to the existing
  96x112 kernel tiling.

dtype follows upstream's compile-time configuration (`ORIG_DTYPE_Q`): the same
single entry source is compiled with `DT_BF16` and with `DT_FLOAT16`. A direct
`ascendc_library` build does not run the OPP tiling-key expander
(`asc_op_compile_base/template_tiling.py`), so the entry carries explicit C++
specializations for exactly the selected 24 combinations per dtype. The SDK
auto-gen then emits one kernel per specialization and the generated
`aclrtlaunch_flash_mla(...)` / `aclrtlaunch_flash_mla_fp16(...)` headers expose
the matching template specializations. The BF16 entry keeps the upstream name
`flash_mla`; FP16 uses `flash_mla_fp16` only because both device binaries are
linked into the single `libsgl_kernel_npu_flash_mla.so` host DSO.

The host selects the specialization per call from the query layout, KV layout,
mask flag and dtype. The template key is owned by the selected SDK launch
specialization and travels with the queued `OpCommand::RunOpApi` call; no
mutable thread-local key is read at execution time. The supported matrix is
BF16/FP16 x 4 layouts x 3 PA KV layouts x mask/no-mask = 48 variants.

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

That target builds the two single-entry AscendC kernel libraries
(`flash_mla_bf16_kernel`, `flash_mla_fp16_kernel`) from the same source.

Metadata generation is package-owned and built from
`csrc/flash_mla_metadata`: an AICPU producer (`libsgl_flash_mla_metadata_aicpu.so`
plus its JSON descriptor) is loaded through ACL CPU kernel mode and the host
entry is assembled into this companion library. No `ASCEND_CUSTOM_OPP_PATH`,
`LD_LIBRARY_PATH` vendor entry, OPP registration, or `cann_ops_transformer`
Python package is required. Sequence lengths are read from device tensors on
every launch and graph replay.

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
ASCEND_RT_VISIBLE_DEVICES=7 python3 -m pytest -q csrc/flash_mla/tests csrc/flash_mla_metadata/tests
```

`LICENSE` and the source headers retain their CANN Open Software License
notices. `SOURCE_ORIGIN.json` records the baseline commit, old/new paths and
whether a file only changed include paths or carries direct-invocation glue.
