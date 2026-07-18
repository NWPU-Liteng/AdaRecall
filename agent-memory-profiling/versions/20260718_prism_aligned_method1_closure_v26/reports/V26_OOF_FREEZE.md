# Method 1 v26 OOF Freeze

Status: `OOF_QWEN_GATE_PASSED_FROZEN_FOR_FRESH_HOLDOUT`

Scientific scope: MedMemoryBench train strict five-fold OOF only. The revealed
v24 test and selection-used v25 development splits were not read or rerun.

## Selected method

`Lexical-C (32) -> local Qwen3-Reranker-0.6B -> Top-5`.

Lexical-C was selected because it achieved the highest OOF Qwen nDCG among
the safe 32-candidate methods. Adaptive Rule also passed every gate, but was
inferior to Lexical-C and is retained as a positive ablation, not promoted.

## OOF result (1,355 queries)

- Lexical-C nDCG@10: `0.116775`.
- vs Dense-C nDCG: `+0.105000`, 95% CI `[0.091060, 0.119181]`.
- vs Full-N nDCG: `+0.108189`, 95% CI `[0.094790, 0.121995]`.
- frontier candidate recall: `0.603752`.
- vs Dense-C frontier recall: `+0.560455`, 95% CI `[0.534563, 0.585979]`.
- final evidence-group coverage: `0.263038`.
- vs Dense-C final coverage: `+0.235855`, 95% CI `[0.212729, 0.258919]`.
- five fold nDCG deltas vs Dense: `+0.0942`, `+0.1180`, `+0.1354`, `+0.1016`, `+0.0762`.
- estimated rerank latency: `1444.56 ms/query` vs Full-N `5778.23 ms/query`;
  estimated net gain `4333.67 ms/query`.
- candidates: 32 vs 128, reduction `75%`.
- namespace, future-memory, duplicate violations: all `0`.

## Negative and bounded results

- Original hypergraph-primary relevance was already falsified by v24.
- Repaired semantic tail remained negative: recall delta `-0.002214`, CI
  `[-0.006273, 0.001845]`.
- OOF-safe history tail was inconclusive and hurt group coverage; its recall
  delta was `+0.003321`, CI `[-0.001971, 0.008733]`.
- Adding semantic/history to adaptive lexical sharply reduced recall relative
  to adaptive lexical alone (`-0.210271`).
- OOF logistic tail selection was weaker than the fixed audit rule.
- Adaptive Rule passed all gates but underperformed Lexical-C (nDCG `0.095831`
  vs `0.116775`), so dynamic gating did not earn final promotion.

Agent replay passed with zero safety violations. The 100-insert incremental
stress passed append-only, normalization, prefix-preservation, and duplicate
rejection checks.

This is not an independent final-test claim. LoCoMo remains unread at this
freeze point.
