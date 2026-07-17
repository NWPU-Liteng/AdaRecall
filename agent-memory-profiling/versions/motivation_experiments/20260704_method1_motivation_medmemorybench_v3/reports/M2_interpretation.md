# M2 Interpretation

M2 formal completed with 960 invocation rows: 240 formal queries across frontier budgets 8, 16, 32, and 64.

## Main Result

| Frontier budget | Selection ms | Rerank ms | Evidence Recall@5 | Evidence Completion@5 | RerankerOracleOverlap@5 |
|---:|---:|---:|---:|---:|---:|
| 8 | 58.55 | 43.94 | 0.110 | 0.000 | 0.554 |
| 16 | 97.91 | 84.51 | 0.110 | 0.000 | 0.720 |
| 32 | 143.77 | 130.30 | 0.112 | 0.000 | 0.858 |
| 64 | 229.23 | 215.75 | 0.111 | 0.000 | 1.000 |

## Supported Claim

Flat frontier expansion increases selection/reranking cost and improves oracle overlap. This supports the motivation that preserving candidate coverage by expanding a flat frontier has a measurable cost.

## Claim Boundary

Do not claim answer quality improves with frontier budget. The answer-generation outputs are noisy and sometimes off target. The evidence metrics use a conservative session-level anchor, so Evidence Completion@5 is zero and Evidence Recall@5 is low and mostly saturated.
