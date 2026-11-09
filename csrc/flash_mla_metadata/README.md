# FlashMLA metadata companion

Package-owned metadata producer for FlashMLA. It is independent of the
FlashMLA compute companion:

- `flash_mla_metadata.cpp` — host operator (`npu::flash_mla_metadata`), built as
  an OBJECT library and assembled into the existing FlashMLA wheel host
  library.
- `aicpu/` — the AICPU metadata kernel, the SectionStreamK scheduling closure,
  the metadata layout header, and a CPU-kernel JSON descriptor. The kernel is
  cross-built with the CANN AICPU toolchain into
  `libsgl_flash_mla_metadata_aicpu.so` and loaded at runtime through ACL CPU
  kernel mode (`aclrtBinaryLoadFromData`, `aclrtRegisterCpuFunc`,
  `OpCommand::RunOpApi`). No OPP registration, `ASCEND_CUSTOM_OPP_PATH`, or
  vendor `libcust_opapi.so` is used.
- `flash_mla_metadata_api.h` — the only shared FlashMLA header between this
  companion and the compute companion (platform core detection and the host
  entry declaration). The AICPU layout header includes the upstream FlashMLA
  `flash_mla_shared.h` layout constants.

## Algorithm and layout

The scheduling algorithm is the upstream AICPU FlashMlaMetadata kernel:
`load_balance::SectionStreamK` with 96x112 tiles, section splitting, FA/FD
layout, mask/window/layout branches, and the same int32 metadata layout consumed
by the FlashMLA compute kernel. The SectionStreamK closure
(`aicpu/load_balance/`, five headers) is copied verbatim from the canonical
origin (see `SOURCE_ORIGIN.json` in the task report). All sequence lengths are
read from device memory inside the AICPU kernel on every launch and every graph
replay; no host D2H copy, `.item()`, or capture-time metadata cache is used.

## Build

A standard A5 build (`SGL_KERNEL_ENABLE_A5_ONLY_OPS`) builds:

- `sgl_kernel_npu_flash_mla_metadata_host` (CMake OBJECT library) assembled into
  `libsgl_kernel_npu_flash_mla.so`;
- `sgl_flash_mla_metadata_aicpu` producing
  `libsgl_flash_mla_metadata_aicpu.so` and its `.json` descriptor, staged in the
  wheel `lib/` directory next to the host library.

The host library locates the AICPU library relative to its own loaded image, so
the wheel works without environment variables or absolute build paths.

## Tests

`tests/test_metadata_graph.py` is vendor-free and covers layouts (BSND/BNSD/
TND), masks 0/3/4 and windows, multi-section prefill, exact metadata structure,
metadata graph replay with changed device lengths (including a zeroed captured
output), a small metadata+compute graph short->long->short, shape rejection,
and fake-tensor shape inference.

```bash
python3 -m pytest -q csrc/flash_mla_metadata/tests
```
