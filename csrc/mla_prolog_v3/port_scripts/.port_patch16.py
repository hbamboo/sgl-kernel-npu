#!/usr/bin/env python3
"""Update README-port.md with the roofline analysis measured in this round."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, "csrc/mla_prolog_v3/README-port.md")

OLD = """## Performance (Ascend950DT, practical peak BW 3.27 TB/s)

| T | latency | effective BW | GEMM throughput |
|---|---|---|---|
| 1 | 48.5 us | 357 GB/s (11% peak) | 0.36 TFLOP/s |
| 8 | 48.1 us | 364 GB/s (11%) | 2.9 TFLOP/s |
| 256 | 96.0 us | 256 GB/s (8%) | 46.1 TFLOP/s |
| 2048 | 688 us | 109 GB/s (3%) | 51.5 TFLOP/s |

Host cost 24 us/call (was 49, then 27 after the first cache). Decode is weight-bandwidth bound: 16.5 MiB bf16 at
3.27 TB/s implies a ~5 us floor; native MXFP8 weights (8.25 MiB) would halve it."""

NEW = """## Performance (Ascend950DT) and roofline

Measured machine limits on this card:

* HBM bandwidth (256 MiB copy): **3.27 TB/s**
* bf16 matmul (4096^3): **424 TFLOP/s**
* machine balance: 424 / 3.27 = **130 FLOP/byte**
* **framework floor**: any NPU kernel op costs **23.4 us** per call on this stack
  (measured with `helloworld` and a tiny `qkv_rms_norm_rope_cache`); the whole
  torch dispatcher + `OpCommand::RunOpApi` + launch path is included.  Our op adds
  ~0.7 us on top of that.

Per call the op streams 17.3 MB of bf16 weights and does 17.3 MFLOP per token, so
it is memory bound up to T ~ 130 and compute bound above.

| T | total | kernel (total - 23.4) | roofline floor | kernel / floor |
|---|---|---|---|---|
| 1 | 45.1 us | 21.7 us | 5.3 us (BW) | 24% |
| 8 | 45.7 us | 22.3 us | 5.4 us (BW) | 24% |
| 256 | 95.8 us | 72.4 us | 10.4 us (compute) | 14% |
| 2048 | 688 us | 665 us | 83.6 us (compute) | 13% |

Core-count sweep (tuning knob `MLA_PROLOG_V3_CORES`, default 32):

| cores | T=1 | T=8 | T=256 | T=2048 |
|---|---|---|---|---|
| 32 | 44.9 | 44.8 | 95.8 | 688.7 |
| 16 | 42.0 | 41.4 | 111.7 | 858.5 |
| 8 | 44.4 | 48.3 | 137.1 | 1065 |
| 4 | 56.9 | 52.7 | 244 | 1922 |

32 cores stays the default (best for compute-bound); 16 is marginally better for
pure decode.

Reading of the remaining gap:

* the four GEMMs are GEMV-shaped at decode (M = tokens), so each of the 32 cores
  streams only ~1/32 of the weights with small DMA requests -> ~25% of peak BW;
* at large T the tiling (sized for the DeepSeek geometry: He=7168, Hcq=1536, N=32)
  gives each core only 24 columns of Hcq=768, so the cube is under-utilised -> ~13%;
* the host side is already at the framework floor, so the only way to remove those
  23.4 us in production is aclgraph capture (SGLang does this)."""

with io.open(P, encoding="utf-8") as f:
    s = f.read()
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW)

OLD2 = """* [ ] kernel-side performance: ~21 us at T=1 vs a ~5 us bandwidth floor"""
NEW2 = """* [x] performance baseline + roofline: host at framework floor (23.4 us), kernel at
      24% (decode) / 13% (prefill) of the measured roofline
* [ ] kernel-side performance work: fp8/MXFP8 weight streaming (halves the bytes) or
      a GLM-specific retiling (32 cores leave only 24 columns per core)"""
assert s.count(OLD2) == 1
s = s.replace(OLD2, NEW2)
with io.open(P, "w", encoding="utf-8") as f:
    f.write(s)
print("readme perf section updated")
