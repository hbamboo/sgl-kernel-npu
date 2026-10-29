# Round 10: actual BF16 prefill on physical device 7

Implementation commits: kernel repository 85309c8 (plus 45071ed / 2bf9753),
SGLang repository 2e06d20952.

Status: experimental prefill path, opt-in in SGLang. The 32768-token result is
about 1.58 ms in a graph benchmark and 1.64 ms in the framework device timeline.
**A complete 32768-token BF16 prologue below 1 ms has not been demonstrated.**

## Actual input and measurement scope

Server: root@113.46.12.25, container hxd-test. Only physical NPU 7 was used.
The original run_mix.sh sets max-prefill-tokens=32768 but leaves the default
chunked-prefill-size=8192. The controlled runner explicitly sets both to 32768.
Two requests of 16384 tokens were scheduled together, with no cached tokens.
Layer 2 inputs were saved from the live framework before operator dispatch.

The checkpoint name contains MXFP8, but these operator inputs are unquantized:

| Tensor | Shape | Dtype / format |
| --- | --- | --- |
| token_x | [T,2048] | BF16 ND |
| weight_dq | [2048,768] | BF16 NZ |
| weight_uq_qr | [768,5120] | BF16 NZ |
| weight_dkv_kr | [2048,576] | BF16 NZ |
| weight_uk | [20,192,512] | BF16 ND |
| gamma_cq / gamma_ckv | [768] / [512] | BF16 |
| sin / cos | [T,64] | BF16 |

The framework dequantizes checkpoint weights once during preparation.
weight_quant_mode=0, kv_cache_quant_mode=0, query_norm_flag=True.
This experiment adds **no activation or weight quantization**. Earlier FP8
experiments are not evidence for this BF16 target.

## Implementation

- Preserve the compiled operator and its small-token/decode dispatch.
- Prepare BF16 projection weights once, with QR/KR column reordering for native
  half-mode RoPE. Keep all intermediate tensors in BF16.
- Use native dense BF16 GEMMs and transpose-batch-matmul for the large prefill.
- Overlap independent RMSNorm/RoPE work on one auxiliary stream shared by all
  layers. Events establish dependencies; record_stream protects allocator
  lifetimes. Return contiguous tensors for the attention/cache ABI.
- SGLang's optional branch requires extend mode, T>=1024, unified unquantized
  KV storage and the exact GLM geometry. Decode never enters this branch.
- Land KV/KR directly into the unified pool, avoiding throw-away scratch caches.

The new implementation is a composition of native kernels, not a new single
fused AICore kernel. Weight preparation is outside steady-state measurements.

## BF16 performance

Captured actual T=32768 input, and its first 8192 rows:

| Tokens | Compiled v3 BF16 | Prepared BF16 composition | Speedup |
| --- | ---: | ---: | ---: |
| 8192 | 2591.713 us | 406.224 us | 6.38x |
| 32768 | 10333.379 us | 1575.584 us | 6.56x |

Method: 5 warmups, graph containing 5 full calls, 21 pairs of NPU timing events,
median duration divided by 5. All five returned outputs are computed. The
microbenchmark excludes unified-pool landing, which the framework includes.

Real framework profiling with two 16K requests and all 47 layers:
- Compiled raw BF16 operator: median 10332.317 us (47 calls).
- Candidate complete annotated prologue plus unified KV landing: median device
  span 1641.654 us, min 1625.311 us, max 2056.758 us (47 calls, 13 core kernels
  per call). Core-duration sum has median 1843.064 us because streams overlap.
- Profile request uses record_shapes=False and with_stack=False. Host annotation
  duration and sums including event waits are not operator device latency.

The work is approximately 566.936 GFLOP per 32768-token call. The measured
large BF16 GEMM peak was approximately 430.61 TFLOP/s, giving a 1.32 ms compute
estimate. This is an empirical comparison, not a proof of a hardware lower bound.

## Precision and application behavior

Full outputs of the actual 32768-token input versus compiled BF16 v3:

| Output | Cosine | Relative L2 | Max absolute difference |
| --- | ---: | ---: | ---: |
| q | 0.999999997475 | 0.000071067 | 0.00390625 |
| qr | 0.999999999430 | 0.000033772 | 0.0625 |
| q_norm | 0.999999999888 | 0.000014947 | 0.0078125 |
| KV | 0.999999999916 | 0.000012991 | 0.015625 |
| KR | 1.0 | 0 | 0 |

A separate staged FP64 CPU golden, with BF16 rounding at intermediate boundaries,
was evaluated on 129 sampled rows including the sequence boundary. Both raw and candidate had similar errors;
q cosine was 0.999999998076 and relative L2 0.000062027. This test does not replace
the full-output raw/candidate comparison above.

Greedy generation is repeatable within each version, but is not bit-exact
between versions. Explicit stream synchronization and contiguous outputs did
not remove the differences. Do not describe end-to-end output equality as passed.

Acceptance uses operator numerical error and model-quality metrics; greedy
generation bitwise equality is not required.

Teacher-forced check on 64 strided real CHAT inputs, last up to 256 tokens each:
16320 valid log probabilities, baseline mean NLL=12.6444216204, candidate
NLL=12.6447138842, delta=+0.000292264. Relative perplexity change=+0.02923%.
Next-token choices match in 58/64 cases. Per-case mean NLL delta ranges from
-0.03984 to +0.03726; mean absolute token-logprob delta is 0.2030. This small
sample checks numerical stability, and does not certify model task quality.

## Existing compiled operator acceptance and decode

The clean library build passed all 27 existing acceptance checks, including
CPU golden, real weights at layers 0/2/23/46, token boundaries, vendor comparison,
output guards, multi-stream and graph tests. Logs: /tmp/mla_v3_round10/acceptance.

Explicit MODE=bf16 baseline/candidate sweeps cover every token count 1 through 92.
All returned outputs and written KV/KR rows are bit-exact. Candidate/baseline
latency ratio: median 0.99665, largest single-run ratio 1.02371. Selected points:

| Tokens | Baseline us | Candidate us |
| --- | ---: | ---: |
| 1 | 90.909 | 90.605 |
| 32 | 94.711 | 96.522 |
| 64 | 105.812 | 103.971 |
| 92 | 117.270 | 118.199 |

A reverse-order repeat at T=1/32/44/64/92 (five timings per size) gave median
ratios 1.00767/0.98930/1.01843/0.99774/0.99494 respectively. This supports
no systematic regression; the largest observed point remains about +2%.

The earlier MXFP8 decode sweep also checked all 92 sizes bit-exact, with median
ratio 0.99341 and largest ratio 1.01445. These are separate dtype experiments.
The new BF16 prefill branch does not change decode dispatch.

## Original benchmark client: small controlled run

Same bench_rate_npu_sv_genedata.py used by bench_glm.sh, rate=26, num-prompts=64,
seed=42, dataset=CHAT-mxfp8-customer-2000.jsonl, port=19002, npu-ids=7+0.
Both servers use --disable-radix-cache, max-running-requests=92,
cuda-graph-max-bs=92, mem-fraction-static=0.85 and chunked-prefill-size=32768.

| Metric | Raw BF16 v3 | Candidate |
| --- | ---: | ---: |
| Request throughput (req/s) | 6.09 | 8.18 |
| Total token throughput (token/s) | 23894.59 | 32084.80 |
| Mean TTFT (ms) | 4654.17 | 3596.75 |
| Mean E2E latency (ms) | 9354.68 | 6668.92 |
| Mean TPOT (ms) | 167.88 | 109.72 |

Total throughput improves by about 34%. TPOT includes interference from prefill;
this does not imply the standalone decode kernel changed by the same amount.
This is not the full 2000-prompt, all-rate production benchmark. Its monitoring
helper npu_stats.py was absent, so the client skipped NPU usage statistics.

## Original benchmark client: full 2000 prompts at rate 26

Both runs completed all 2000 requests successfully. Input tokens=7566887 and
output tokens=57990 in each run. No cache hits were used. Both runs reached
92 concurrent requests, and each had 264 prefill steps including 214 steps
with exactly 32768 new tokens. The request input distribution was identical.

| Metric | Raw BF16 v3 | Candidate |
| --- | ---: | ---: |
| Duration (s) | 321.866 | 246.314 |
| Request throughput (req/s) | 6.214 | 8.120 |
| Total token throughput (token/s) | 23689.634 | 30955.970 |
| Output throughput (token/s) | 180.168 | 235.432 |
| Mean TTFT (ms) | 122589.024 | 81837.986 |
| Mean E2E latency (ms) | 129890.12 | 86864.58 |
| Mean TPOT (ms) | 260.806 | 179.557 |

Total throughput improves by 30.68%; run duration decreases by 23.47%.
At offered rate 26, both single-card servers accumulate a queue; these TTFT/E2E
numbers include queue time. Rates 28 through 40 were not run in this comparison.
The server processes started by this test were terminated after completion.

Run the same controlled raw/candidate pair on an idle device 7:

```bash
OUT=/tmp/new_bf16_bench python3 csrc/mla_prolog_v3/tests/e2e/bench_glm_7.py
```

## Reproduction

From /mnt/paas/hxdnpu/sgl-kernel-npu inside hxd-test:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
cmake -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DASCEND_INCLUDE_DIR=/usr/local/Ascend/cann-9.1.0/aarch64-linux/include \
  -DSOC_VERSION=Ascend950PR_9599 -DBUILD_DEEPEP_MODULE=OFF \
  -DBUILD_KERNELS_MODULE=ON -B /tmp/bk-round10 -S .
cmake --build /tmp/bk-round10 --target sgl_kernel_npu -j16
OUT=/tmp/mla_v3_round10/acceptance bash csrc/mla_prolog_v3/tests/run_acceptance.sh 7

CASE=/tmp/mla_v3_round10/framework_baseline32/T32768.pt TOKENS=8192,32768 \
  python3 csrc/mla_prolog_v3/tests/bench_framework_case.py
CASE=/tmp/mla_v3_round10/framework_baseline32/T32768.pt \
  python3 csrc/mla_prolog_v3/tests/check_bf16_prefill_golden.py

MODE=bf16 LIB=/tmp/mla_v3_round10/baseline.so \
  OUT=/tmp/mla_v3_round10/decode_bf16_base TOKENS=$(seq -s, 1 92) \
  python3 csrc/mla_prolog_v3/tests/bench_pipeline.py
MODE=bf16 REF=/tmp/mla_v3_round10/decode_bf16_base \
  OUT=/tmp/mla_v3_round10/decode_bf16_candidate TOKENS=$(seq -s, 1 92) \
  python3 csrc/mla_prolog_v3/tests/bench_pipeline.py
```

Apply tests/e2e/sglang-bf16-prefill.patch to the matching SGLang checkout, or use
its corresponding Git branch. The BF16 optimization is off unless its flag is
set. In an idle card-7 environment, start one server at a time:

```bash
SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE=1 \
  bash csrc/mla_prolog_v3/tests/e2e/run_prefill_7.sh --disable-radix-cache
# Set the flag to 0 for the raw BF16 v3 control.
# Optional: SGLANG_NPU_MLAPO_V3_CAPTURE_DIR=/tmp/new_capture captures actual inputs.
# Optional diagnostic synchronization: SGLANG_NPU_MLAPO_V3_BF16_DIAG_SYNC=1.
```

The prepared weights and streams are cached per layer/caller. The default runner
uses the candidate for evaluation; production framework default remains off.
All captured inputs, experiment libraries and logs live under
/tmp/mla_v3_round10. No model weights or private dataset content are committed.

## Rejected variants and remaining work

Joint DQ/DKV and merged QC/QR native GEMMs are slower (1.75-1.96 ms for their
complete serialized compositions). Standalone custom BF16 BMM variants are exact
but slower than native transpose-batch-matmul (~440 us): N128/K96 ~498 us,
N256/K64 ~593 us. Unrolling N remains ~496 us; a double-buffered full-A
M80 variant remains ~502 us. Experimental vector fusion was also slower. None is enabled.

Removing the BF16 large-M PIPE_M barrier preserved all actual 32768-token
outputs bit-exact but left the raw latency at 10.34 ms. It was reverted.

Further improvement must address dense BF16 compute and L1/L0 reuse. The raw
NO_QUANT operator uses MlaPrologVecS1CubS2 (Split-N), whereas MXFP8 uses Split-M.
Raw BF16 profiling shows ~19% MAC ratio and ~60% MTE1 ratio; MXFP8-only tiling
changes should not be treated as optimization of the real BF16 prefill path.
The <1 ms target and remaining rate settings remain open. Numerical acceptance
and the sampled model probability metric pass the agreed checks; broader task
quality evaluation is outside the coverage of this sample.
