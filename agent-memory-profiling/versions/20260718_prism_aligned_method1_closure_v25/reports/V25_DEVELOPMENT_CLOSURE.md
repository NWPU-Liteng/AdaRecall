# Method 1 v25 Development Closure

Status: `V25_DEVELOPMENT_CLOSED_AWAITING_FRESH_HOLDOUT`

v25 does not read or rerun the already revealed v24 test split. Results are development-only and require a fresh holdout for an independent final claim.

## Main result

- Frontier candidate recall vs Dense-C: `+0.022414`; 95% CI `[0.005172413793103448, 0.041379310344827586]`.
- Final evidence-group coverage vs Dense-C: `+0.022414`; 95% CI `[0.005172413793103448, 0.041379310344827586]`.
- Final nDCG@10 vs Dense-C: `+0.009011`; 95% CI `[-0.0005536497300124996, 0.02107981790608779]`.
- Final nDCG@10 vs Full-N: `+0.005411`; 95% CI `[0.0, 0.012547058220183498]`.
- Mean latency: v25 `819.02 ms`, Full-N `3120.93 ms`, net gain `2301.90 ms/query`.
- Candidate count: 32 vs Full-N 128 (75% reduction).
- Safety: `PASS`.

## Algorithm conclusion

Degree/overlap calibration removes saturated broad-edge support, but hypergraph relevance still generalizes weakly. The promoted development method uses Dense plus a sparse lexical safety frontier for exact medical entities and dates, followed by the frozen Qwen reranker. Hypergraph evidence remains an auditable diagnostic/layout signal rather than a primary relevance signal.

## Boundaries

- Residual hypergraph replacement failed the held-out development partition and was rejected.
- No new dataset was downloaded.
- No v25 test run exists; a fresh holdout or new benchmark is required before claiming independent closure.
