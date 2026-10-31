# Decode Qn N256 experiment

Keep K64x3 and existing double-buffer synchronization, enlarge Qn N to256 for
M<=64. L0B=32KiB and L0C<=64KiB per ping/pong. Real-weight outputs were bit-exact,
but no material gain remained after the chosen core/groups tuning. Not enabled
in production. A few timing rows show host/device interference; do not use those
rows as performance claims. Reproduction patch and raw measurements retained.
