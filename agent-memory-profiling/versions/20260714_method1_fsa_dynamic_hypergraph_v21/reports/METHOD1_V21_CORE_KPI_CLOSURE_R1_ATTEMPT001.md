# Method 1 v21 Core KPI Closure

This is a descriptive train-OOF closure over the immutable tuned-r2 outputs. No calibration or formal-test input was read, and no minimum-effect threshold is used.

## Metric semantics

- `group recall@K` is the fraction of required evidence groups hit by at least one acceptable memory. It is the task-aligned retrieval recall and reproduces the prior `group_coverage` metric.
- `evidence-member recall@K` is conventional set recall over the union of all acceptable memories. It penalizes failure to retrieve alternatives that are not simultaneously required, so it must not replace group recall.
- `exact task success` is the fraction of queries covering every evidence group. It is a retrieval-set accuracy proxy, not final Agent answer accuracy.
- `logical Tiles` are host-side logical layout counts, not PIM hardware measurements.

## Fixed deployment point: 16 activated prototypes, 32 members, beam 32, B=32

| method | group recall | member recall | member precision | exact success | e2e mean | e2e p95 | payload bytes | logical Tiles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_joint_semantic_bundle_hypergraph | 0.687491 | 0.538553 | 0.040959 | 0.613727 | 5.011 ms | 7.008 ms | 118689.5 | 52.608 |
| ann_flat | 0.685978 | 0.536475 | 0.040752 | 0.613284 | 2.666 ms | 4.098 ms | 118545.5 | 52.608 |
| same_capacity_neural_flat | 0.669459 | 0.505907 | 0.038441 | 0.596458 | 3.718 ms | 5.897 ms | 116316.0 | 52.608 |
| budgeted_pairwise_factor_graph | 0.570012 | 0.389735 | 0.029031 | 0.505978 | 5.810 ms | 7.709 ms | 112501.1 | 52.608 |

## Closure interpretation

The output reports raw paired-method deltas and exact Pareto dominance only. A small positive, zero, or negative effect remains a measured result; it is not converted into a method-valid/method-invalid gate.

The retrieval handoff is the selected B-memory set. `rerank_executed=false`, so no downstream reranker or final Agent answer-accuracy result is claimed. The primary B=32 values are group recall `0.687491`, exact task success `0.613727`, mean e2e `5.011 ms`, and logical Tiles `52.608`.

Dynamic content-update mechanics are implemented separately as the audited static overlay fallback. They do not turn this OOF result into a causal streaming evaluation.
