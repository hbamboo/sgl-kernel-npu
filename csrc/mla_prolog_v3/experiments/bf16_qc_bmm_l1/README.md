# BF16 QC projection + BMM through L1 (scheme 1)

**Experiment only: numerical checks pass, but the fused implementation is slower.**
It is not linked into the production library and is not routed from SGLang.
The existing prepared BF16 prefill and all decode dispatch remain unchanged.

## What was verified

Input: BF16 normalized CQ [T,768], BF16 Wqc [20,768,192], BF16 Wuk [20,192,512].
Output: BF16 Q [T,20,512]. All weights are prepared before steady-state timing.

1. Compute normalized CQ times Wqc into FP32 L0C.
2. Fixpipe casts that result to BF16 directly in NZ layout in L1.
3. Load the BF16 QC from L1 into L0A and multiply by Wuk.
4. Store Q in the normal T/head/512 layout expected by the framework.

`F322BF16` preserves the original BF16 intermediate rounding. Despite the API
field name `quantPre`, this mode is a dtype cast and introduces no FP8/INT8
activation or weight quantization. When debug=False, the QC intermediate never
round-trips through GM, eliminating its approximately 480 MiB of GM read/write
at T=32768. debug=True additionally emits QC only for correctness inspection.

L1 allocation is 192 KiB norm + 64 KiB projection-weight tiles + 192 KiB Wuk
+ 64 KiB QC = 512 KiB. L0A/L0B each use 64 KiB; L0C uses 128 KiB.
Each task processes one head and a 4096-token slice, on 32 AIC cores.

## Results on physical NPU 7, Ascend950PR_9599, CANN 9.1.0

Inputs were taken from the actual framework's captured 32768-token BF16 case.
The first 8192 rows form the smaller test. Kernel-only times are graph medians
(5 warmups, 5 calls per graph, 21 pairs of timing events).

| First projection recipe (M=128) | Order | T8192 fused us | T32768 fused us |
| --- | --- | ---: | ---: |
| N192 / K64 | N outer | 571.429 | 1429.034 |
| N96 / K128 | N outer | 802.017 | 2006.027 |
| N64 / K256 | K outer, A reused over N | 671.196 | 1679.255 |
| Native QC GEMM + transpose BMM | two kernels | 228.867 | 894.789 |

All three recipes produced bit-exact QC and Q for the captured inputs.
Boundary checks also covered T=1/7/92/128/129. An additional 22-size random
BF16 input sweep, including 4095/4096/4097, produced 44 bit-exact QC/Q checks. The K-outer version reduces
reloading A across N but still does not beat the native pair.

The best recipe was inserted into an isolated **full prefill** function,
including DQ, normalization, QR, RoPE, DKV/KR and Q:

| Tokens | Existing BF16 full path us | Fused-L1 full path us |
| --- | ---: | ---: |
| 8192 | 405.806 | 744.258 |
| 32768 | 1573.022 | 2080.903 |

All five full-path outputs (Q, QR, Q_norm, KV, KR) were bit-exact at both sizes.
At T32768 the full path is approximately 32.2% slower. No framework integration
or end-to-end serving run was performed for this rejected candidate.

The best recipe's device profile reports 1427.857 us total, with MAC 862.879 us
(60.5%), MTE1 1082.237 us (75.8%), MTE2 799.382 us (56.0%) and Fixpipe 574.729 us
(40.3%). These stages overlap, so their durations must not be summed.
Reducing GM traffic did not compensate for L1/L0 transfers and transitions
between the two differently shaped GEMMs. The broad claim that any possible
QC/BMM fusion must be slower is not established by these experiments.

Machine-readable measurements and precision summaries are in results.json.

## Build and reproduce

Run in hxd-test, from the repository root; use an idle physical card 7.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
EXP=csrc/mla_prolog_v3/experiments/bf16_qc_bmm_l1
cmake -S "$EXP" -B /tmp/bk-qc-bmm-l1-final \
  -DQC_N=192 -DQC_K=64 -DQC_K_OUTER=0
cmake --build /tmp/bk-qc-bmm-l1-final -j16

export LIB=/tmp/bk-qc-bmm-l1-final/libfused_qc_bmm.so
export CASE=/tmp/mla_v3_round10/framework_baseline32/T32768.pt
STRICT=1 TOKENS=1,7,92,128,129,8192,32768 python3 "$EXP/check.py"
STRICT=1 TOKENS=8192,32768 python3 "$EXP/check_pipeline.py"
```

The other tested CMake recipes are QC_N=96/QC_K=128/QC_K_OUTER=0 and
QC_N=64/QC_K=256/QC_K_OUTER=1. Build them in separate folders and set LIB to
that folder's library; load only one experimental library per Python process.
RANDOM=1 checks random BF16 normalized CQ with the same real weights.
Without STRICT=1, numerical acceptance requires cosine>0.99999 and relative
L2<0.005. Intermediate emission is disabled in every performance measurement.

Logs and first-version source/library snapshots are retained at
/tmp/mla_v3_round11. Production baseline commits are 3ba25e5 in the kernel
repository and 2e06d20952 in SGLang. The experiment uses a separate Git branch.
