# M1 Interpretation

M1 formal completed with 1688 invocation rows and 240/240 formal queries represented.

## Main Result

The full-history selection latency was 132.29 ms for `medical_only` and 134.25 ms for `medical_plus_official_noise`. Host/index/payload bytes increased with larger history scales and with official noise. Frontier fill ratio stayed at 1.0 across reported scales.

## Supported Claim

Longitudinal histories and official distractors increase the amount of memory state that must be represented and materialized for flat retrieval.

## Claim Boundary

Do not claim a universal monotonic latency law from M1 alone. Some partial scales are close or non-monotonic because candidate composition and reranker behavior affect latency.
