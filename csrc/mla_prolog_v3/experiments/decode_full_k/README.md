# Rejected decode Qn full-K experiment

K=192 loaded at once, N=256 L1 panels, N=64 L0 tiles, M<=64.
L1A<=24 KiB, L1B=96 KiB, L0A/L0B<=24 KiB, L0C<=16 KiB per ping/pong.
BF16 intermediate rounding is preserved. All 17 shapes and graph replay checks
were bit-exact, but T17..64 were slower than the accepted packed-output kernel.
T64 packed-only ~31.25 us vs full-K ~34.37 us. The route is removed from production.

Apply experiment.patch on the packed-output implementation, build the library,
then run check.py with source-tree PYTHONPATH and an idle NPU. The checker selects
the original route for references and toggles MLA_PROLOG_V3_DECODE_FULL_K=1 for
the candidate. Each graph contains 47 calls, timed in 31 alternating pairs.
results.json compares original contiguous-output+cat against packed+full-K;
compare the packed-only report for the additional internal-kernel regression.
