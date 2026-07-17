# Final Motivation Summary

Updated: 2026-07-05

## Scope and Claim Boundary

This package evaluates the motivation chain for Method-1 on the English text-only MedMemoryBench workload. It does not evaluate a proposed hypergraph layout, PIM runtime, replication policy, online remapping, or dense-vs-graph-vs-hypergraph method comparison.

Allowed high-level conclusion:

> Longitudinal personal-health memory creates large, noisy candidate histories. Flat frontier expansion increases materialization/reranking cost and improves oracle overlap, but evidence returns are mostly saturated under the current session-level evidence anchor. The selected top memories are scattered across chronological tiles, creating over-fetch headroom. Under a fixed auditable controller, controller-defined execution stages induce statistically non-random stage-conditioned memory-access locality. These findings motivate stage-conditioned structural memory organization, but they do not prove final layout or hardware superiority.

## Data and Provenance

- Dataset: official English MedMemoryBench files under `/data2/user25263093/userliteng/26_code/preTest/data/external/MedMemoryBench`.
- Schema audit: `reports/01_medmemorybench_schema_audit.md`.
- Memory items: 42,034 turn-pair/singleton items in `aggregate/memory_items_turn_pair.jsonl`.
- Evidence mapping: 1,935/1,939 queries recoverable, 99.79% coverage, in `aggregate/query_evidence_mapping.jsonl`.
- Evidence anchor: conservative session-level mapping from query `source_key_points.session_id` to medical memories in that persona/session.
- Dataset caveat: source Git repository revision remains pending because no `external/MedMemoryBench` Git clone was present.

## M1: History-Scale Cost

Artifacts:

- `aggregate/M1_history_scale_invocations.jsonl`: 1688 rows.
- `aggregate/M1_history_scale_summary.csv`: 9 summary rows.
- Formal query coverage: 240/240.

Main observations:

- Medical-only full-history mean selection latency: 132.29 ms.
- Medical-plus-official-noise full-history mean selection latency: 134.25 ms.
- Frontier fill ratio stayed at 1.0 across reported scales.
- Host/index/payload byte footprint grows with larger history scales and with official noise.

Interpretation:

M1 supports the workload premise that histories and distractors increase the amount of memory state that must be represented and scanned/materialized. The latency curve is not strictly monotonic at every scale because reranker and candidate composition effects dominate some settings; report observed component behavior rather than claiming a universal causal law.

## M2: Frontier Budget

Artifacts:

- `aggregate/M2_frontier_budget_invocations.jsonl`: 960 rows.
- `aggregate/M2_frontier_budget_summary.csv`: 24 summary rows.
- `aggregate/M2_frontier_occupancy.csv`: 960 rows.
- Formal query coverage: 240/240.

Budget means:

| Frontier budget | Selection ms | Rerank ms | Evidence Recall@5 | Evidence Completion@5 | RerankerOracleOverlap@5 |
|---:|---:|---:|---:|---:|---:|
| 8 | 58.55 | 43.94 | 0.110 | 0.000 | 0.554 |
| 16 | 97.91 | 84.51 | 0.110 | 0.000 | 0.720 |
| 32 | 143.77 | 130.30 | 0.112 | 0.000 | 0.858 |
| 64 | 229.23 | 215.75 | 0.111 | 0.000 | 1.000 |

Interpretation:

M2 supports a cost/frontier/oracle-overlap motivation: expanding the frontier materially increases selection and reranking latency, while reranker-oracle overlap rises to 1.0 by budget 64. Under the session-level evidence anchor, Evidence Recall@5 is low and mostly flat, and Evidence Completion@5 is zero because the gold session-level set is usually much larger than top-5.

Do not claim that larger frontier budgets improve answer quality. Raw answer generations are noisy and sometimes off target, so answer text is not a strong task-quality signal in this package.

## M3: Chronological Tile Scattering

Artifacts:

- `aggregate/M3_tile_scattering.csv`: 2160 rows.
- Formal query coverage: 240/240.
- Source selections: fresh M2 formal budget-64 top-5 rows.

Mean activated tiles and access amplification:

| Tile capacity | Layout | Activated tiles | Access amplification | Scatter ratio |
|---:|---|---:|---:|---:|
| 8 | chronological | 4.50 | 7.21 | 4.50 |
| 8 | random_per_persona | 4.94 | 7.90 | 4.94 |
| 8 | oracle tile-packing lower bound | 1.00 | 1.60 | 1.00 |
| 16 | chronological | 4.21 | 13.48 | 4.21 |
| 16 | random_per_persona | 4.90 | 15.68 | 4.90 |
| 16 | oracle tile-packing lower bound | 1.00 | 3.20 | 1.00 |
| 32 | chronological | 3.84 | 24.56 | 3.84 |
| 32 | random_per_persona | 4.85 | 31.04 | 4.85 |
| 32 | oracle tile-packing lower bound | 1.00 | 6.40 | 1.00 |

Interpretation:

M3 supports the physical over-fetch premise. Chronological placement scatters the selected top-5 memories across roughly 3.84 to 4.50 tiles on average, while an oracle tile-packing lower bound would use one tile for top-5 at the tested capacities. This quantifies headroom for better structural placement, not superiority of a specific proposed layout.

## O1: Controller-Defined Stage Locality

Artifacts:

- `aggregate/O1_formal_stage_trace.jsonl`: 480 rows.
- `aggregate/O1_same_cross_stage_stats.csv`: 4 rows.
- `aggregate/O1_stage_shuffle_null.csv`: 1000 permutations.
- `aggregate/O1_pair_recurrence.csv`: 4 rows.
- Formal query coverage: 120/120, exactly four stages per query.

Controller quality:

- Parsed intent JSON: 479/480 rows, 99.79%.
- Retrieval-goal prefix validity: 100%.
- Queries with more than one unique selected set across the four stages: 120/120.
- Mean unique selected sets per query: 3.925/4.
- Mean intent generation latency: 7037.96 ms.
- Mean selection latency after intent: 147.57 ms.

Same-stage versus cross-stage overlap:

| Stage | Same-stage overlap | Cross-stage overlap | Same-minus-cross |
|---|---:|---:|---:|
| ground | 0.06099 | 0.04708 | 0.01391 |
| plan | 0.16155 | 0.04155 | 0.11999 |
| reason | 0.05127 | 0.04453 | 0.00675 |
| reflect | 0.03801 | 0.04149 | -0.00347 |

Stage-shuffle null:

- Observed mean same-minus-cross statistic: 0.03429.
- Null mean: -0.00028.
- Null 95% interval: [-0.00545, 0.00536].
- Empirical p-value: 0.000999.
- Standardized effect size: 12.48.

Interpretation:

O1 supports the claim that controller-defined execution stages induce statistically non-random stage-conditioned memory-access locality. The strongest contribution comes from the Plan stage; Ground and Reason are positive but smaller, while Reflect is slightly negative in same-minus-cross overlap. The careful wording is therefore: "the fixed controller induces measurable stage-conditioned locality, especially for planning-stage retrieval." Do not claim that every stage independently has strong positive locality.

## Figures

- `figures/Fig_M1_history_scale_latency.png`
- `figures/Fig_M1_history_scale_traffic.png`
- `figures/Fig_M2_frontier_cost_evidence.png`
- `figures/Fig_M2_budget_oracle_overlap.png`
- `figures/Fig_M2_budget_task_quality.png`
- `figures/Fig_M3_tile_utilization.png`
- `figures/Fig_M3_access_amplification.png`
- `figures/Fig_M3_scatter_ratio_by_query_type.png`
- `figures/Fig_O1_same_vs_cross_stage_overlap.png`
- `figures/Fig_O1_stage_shuffle_null.png`
- `figures/Fig_O1_stage_pair_recurrence.png`

## Validation

Final validation passed:

- `reports/VALIDATION.md`: PASSED.

## Remaining Method-Evaluation Work

- Dense-vs-graph-vs-hypergraph comparisons.
- PIM runtime measurement or simulation.
- Structural placement, replication, and online remapping evaluation.
- Statistical testing for M1/M2 slopes or Pareto tradeoffs if the paper needs stronger inferential claims.
