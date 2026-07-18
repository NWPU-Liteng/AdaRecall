# Method 1 v26 Final Closure

Status: `V26_OOF_SUCCESS_FRESH_RUN_FAILED_AWAITING_NEW_HOLDOUT`

## What was falsified

- Hypergraph relevance cannot replace Dense relevance (v24 test failure).
- Repaired semantic support stayed negative in OOF.
- OOF-safe `co_candidate` history was inconclusive and is not native Agent access.
- Semantic/history fusion reduced adaptive lexical recall by `-0.210271`.
- OOF logistic tail selection was weaker than the audit rule.
- Adaptive Rule passed OOF, but did not beat the simpler Lexical-C control.

## What passed OOF

The selected method is Lexical-C (32 candidates) followed by the frozen local
Qwen reranker. On 1,355 strict five-fold OOF queries:

- nDCG@10 `0.116775`.
- nDCG vs Dense-C `+0.105000`, 95% CI `[0.091060, 0.119181]`.
- nDCG vs Full-N `+0.108189`, 95% CI `[0.094790, 0.121995]`.
- frontier recall `0.603752`; vs Dense-C `+0.560455`, CI `[0.534563, 0.585979]`.
- final group coverage `0.263038`; vs Dense-C `+0.235855`, CI `[0.212729, 0.258919]`.
- 5/5 fold nDCG deltas were positive.
- latency estimate `1444.56 ms/query` vs Full-N `5778.23 ms/query`;
  net gain `4333.67 ms/query`.
- candidate reduction `75%`; all safety violations `0`.

Adaptive Rule also passed all gates: nDCG vs Dense-C `+0.084056`, CI
`[0.071992, 0.096601]`. It was not selected because Lexical-C was stronger.

## Fresh holdout

LoCoMo is genuinely independent of MedMemoryBench: different authors,
conversations, distribution, and annotations. Config `8a8d55b`, manifest, and
read lock commit `7bbca88` existed before the first gold read.

The only formal read failed before metrics because the released evidence field
contained an undocumented composite string (`D8:6; D9:17`). The receipt was
created before parsing, so the read is consumed. No split-and-rerun fix was
applied. Therefore there is no independent quality result and no final success
claim; a new unread holdout is required.

Agent replay and the 100-insert incremental stress passed. v26 tests: 4 passed;
compileall and JSON validation passed.
