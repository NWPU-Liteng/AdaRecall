# O1 Interpretation

O1 formal completed with 480 stage rows: 120 formal queries and four controller-defined execution stages per query.

## Controller Quality

- Parsed intent JSON: 479/480 rows, 99.79%.
- Retrieval-goal prefix validity: 100%.
- Queries with more than one unique selected set across the four stages: 120/120.
- Mean unique selected sets per query: 3.925/4.
- Mean intent generation latency: 7037.96 ms.
- Mean selection latency after intent generation: 147.57 ms.

## Stage-Locality Result

| Stage | Same-stage overlap | Cross-stage overlap | Same-minus-cross |
|---|---:|---:|---:|
| ground | 0.06099 | 0.04708 | 0.01391 |
| plan | 0.16155 | 0.04155 | 0.11999 |
| reason | 0.05127 | 0.04453 | 0.00675 |
| reflect | 0.03801 | 0.04149 | -0.00347 |

The 1000-permutation stage-shuffle null gives:

- Observed statistic: 0.03429.
- Null mean: -0.00028.
- Null 95% interval: [-0.00545, 0.00536].
- Empirical p-value: 0.000999.
- Standardized effect size: 12.48.

## Supported Claim

The fixed controller induces measurable non-random stage-conditioned memory-access locality. The strongest stage contribution is planning-stage retrieval.

## Claim Boundary

Use `controller-defined execution stage`. Do not describe O1 labels as source-provided execution-stage labels. Do not claim every stage independently has strong positive locality: Reflect is slightly negative in same-minus-cross overlap, and the aggregate effect is dominated by the Plan stage.
