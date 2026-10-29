# Independent MLA prolog prefill with FIA: implementation and A/B

The prefill attention backend can now be changed independently of MLA prolog.
A uses FlashMLA prefill and B uses FIA prefill. Both retain the same prepared
BF16 prologue and FlashMLA decode. Default routing remains unchanged.

Implementation: SGLang commit 3dc7468816, kernel test/scripts commit 4be3263.
Only physical device 7 was used, in hxd-test on root@113.46.12.25.

## Routing and configuration

- New flag SGLANG_NPU_USE_MLAPO_V3_PREFILL=1 selects absorbed MLA preprocessing
  for ordinary prefill even when ENABLE_ASCEND_FLASH_MLA_PREFILL=0.
- FlashMLA prefill retains priority when ENABLE_ASCEND_FLASH_MLA_PREFILL=1.
- With the independent flag off, FlashMLA-off prefill retains its original
  MHA/DSA dispatch. The decode branch is unchanged.
- The new FIA path requires V3 prolog, unquantized unified KV, rank 512 and RoPE
  dimension 64. Validation covers dense GLM with TP=1 and regular prefill.
- The service runner now honors an externally supplied FlashMLA prefill flag.
  Its global ENABLE_ASCEND_FLASH_MLA remains 1, preserving FlashMLA decode.

Start FIA prefill with MLA prolog, in hxd-test:

```bash
ENABLE_ASCEND_FLASH_MLA_PREFILL=0 \
SGLANG_NPU_USE_MLAPO_V3_PREFILL=1 \
bash /mnt/paas/hxdnpu/sgl-kernel-npu/csrc/mla_prolog_v3/tests/e2e/run_prefill_7.sh \
  --disable-radix-cache
```

Change ENABLE_ASCEND_FLASH_MLA_PREFILL to 1 for the FlashMLA control. The runner
uses physical NPU 7, localhost:19002, max running requests / decode graph BS 92,
context length 32768, max-prefill-tokens=32768 and chunked-prefill-size=32768.
SGLANG_NPU_MLAPO_V3_PREFILL_BF16_NATIVE remains 1 in both variants.

## FIA adaptation

FIA's paged MLA implementation in CANN 9.1 rejects 20 query heads and strided
key_rope tensors. The unified pool interleaves 512 latent + 64 RoPE values per
token, so its two views cannot be passed directly to this interface.

The adapter computes the valid page list and compact block table once per
prefill step. Each layer gathers only referenced pages from the contiguous
unified pool, then makes contiguous latent/RoPE buffers for FIA. It never makes
the entire KV pool's split views contiguous. Prefixes, chunked prompts, zero
query-length requests and query padding are handled explicitly.

Serving pads the 20 query heads to 32 for the FIA call, then discards unused
heads. Splitting the heads into 16+4 was tested and rejected:

| Total query tokens | Pad to 32 us | Split 16+4 us |
| --- | ---: | ---: |
| 8192 | 6639.221 | 13794.679 |
| 32768 | 99916.626 | 221278.152 |

These measurements use actual captured BF16 prologue outputs and two equal
sequences with no prefix. The two head-handling variants were bit-exact. The
split option remains available only as an internal experiment; serving defaults
to padding. Both methods introduce no activation or weight quantization.

## Functional and numerical checks

The standalone tests cover five dispatch combinations, nonzero storage offsets,
no prefix, prefixes, page boundaries, zero query lengths, 20/32/1 heads and query
padding. FIA was checked against a CPU FP32 causal attention golden. Across eight attention cases, minimum cosine was 0.999997657 and maximum
relative L2 was 0.002165, within cosine>0.99999 / relative L2<0.005. This
includes query length 2051 with KV length 4097, crossing the compressed causal
mask boundary at 2048. These are attention-adapter checks; prolog math is unchanged.

The service smoke test completed two 16384-token prompts together (32768 new
tokens), followed by eight decode tokens each. Two follow-up 16512-token prompts
hit 16384 cached tokens each and completed decoding successfully. All 47 layers
logged BF16 prolog prefill with quant_mode=0; decode retained FlashMLA.

## Controlled serving test

Client: the same bench_rate_npu_sv_genedata.py invoked by bench_glm.sh, rate 26,
2000 prompts, seed 42, original CHAT-mxfp8-customer-2000.jsonl dataset. Both runs
use --disable-radix-cache and mem-fraction-static=0.85. Before the timed benchmark,
each variant runs the same 64-input teacher-forced probability check as warmup.
Startup and quality-check times are excluded from benchmark duration.

Both variants completed all 2000 requests, with exactly 7,566,887 input tokens
and 57,990 output tokens each. One sequential A-then-B run was measured.

| Metric | A: FlashMLA prefill + prolog | B: FIA prefill + prolog |
| --- | ---: | ---: |
| Duration (s) | 245.568 | 438.945 |
| Request throughput (req/s) | 8.144 | 4.556 |
| Output throughput (tok/s) | 236.147 | 132.112 |
| Total throughput (input + output tok/s) | 31,050.025 | 17,370.928 |
| Mean TTFT (ms) | 84,956.350 | 184,847.304 |
| Mean TPOT (ms) | 179.566 | 302.187 |

B has **44.055% lower total throughput** in this configuration. Keep A as the
recommended setting for this workload; the independent FIA path works but is
an opt-in alternative. No new quantization is introduced, and this change does
not reduce the prolog's own approximately 1.58 ms cost toward the 1 ms goal.

The separate quality check used 64 cases and 16,320 teacher-forced logprobs:
A mean NLL = 12.6447138842, B = 12.6447009955, delta = -0.0000128887;
B/A perplexity change = -0.00128886%. The next generated token matched in
56/64 cases; token identity was not the agreed acceptance criterion. These
sampled quality metrics show no observed regression, rather than establishing
quality equivalence on every input.


TTFT/TPOT include interference and queueing at the offered rate. The comparison
measures the complete attention backend adaptation, including FIA's page packing
and required head padding. It does not isolate only the two vendor kernel bodies.
The standalone decode implementation is unchanged; mixed-load decode latency can
change with prefill compute contention.

## Reproduction and artifacts

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
cd /mnt/paas/hxdnpu/sgl-kernel-npu
python3 csrc/mla_prolog_v3/tests/e2e/test_fia_mla_prefill.py
OUT=/tmp/new_attention_ab NUM_PROMPTS=2000 REQUEST_RATE=26 \
  python3 csrc/mla_prolog_v3/tests/e2e/bench_prefill_attention_ab_7.py
```

The A/B runner starts only one server at a time and terminates only its own child
servers. It verifies request completion and equal input/output token counts, and
saves metrics and probability comparisons in summary.json. It refuses to start
when the chosen port is occupied. The port can be overridden with PORT.

This run's logs and raw benchmark output are retained under:
/tmp/mla_v3_round12/ab/20261029_123116.
The compact metrics and all eight numerical checks are versioned in
results-prefill-fia-ab-20261029.json beside this report.
Small API probes, standalone numerical tests and smoke results live under
/tmp/mla_v3_round12. Model weights and private input content are not committed.
