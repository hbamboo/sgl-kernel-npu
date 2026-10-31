# Rejected wide-L0B Qn route

Full K192 and N128 use 48KiB of L0B with a single L0B buffer. L1 panels are N256.
All real-weight outputs in the nine decode sizes were bit-exact. No repeatable
improvement over packed Query plus selected core groups was measured; removed.
experiment.patch preserves the candidate, results.json records all comparisons.
The raw static sync audit emitted unresolved cross-function pairing candidates;
these are retained in the task's sync-pair-candidates.json rather than cleared.
