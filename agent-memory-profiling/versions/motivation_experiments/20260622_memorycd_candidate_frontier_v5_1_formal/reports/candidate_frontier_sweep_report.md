# Candidate Frontier Sweep Report

Summary CSV: `versions/20260622_memorycd_candidate_frontier_v5_1_formal/reports/candidate_frontier_sweep_summary.csv`

This report summarizes the V5.1 selection sweep traces. Full Qwen task-quality curves are intentionally not inferred from selection-only runs.

## Current Run

- Protocols: `official_single`, `official_cross`, `causal_cross`
- Tasks: `rating_prediction`, `item_ranking`
- Events per protocol: 1000
- Trace rows per protocol: 11000
- Total trace rows: 33000
- Summary rows: 264
- Matrix: `R in {32,64,128}`, `B in {8,16,32,64}` with `B <= R`, `K=8`
- Retrieval audit: `faiss_flat` validated with Top-R score-boundary tolerance `1e-6`
- Reuse policy: query encoded once per event; MiniLM rerank reused per `B`

## Scope Note

This is a cached selection-only run over the first 1000 eligible rating/ranking events per protocol, not the full 15504-event protocol sweep and not Qwen task-quality evaluation. `t_query_encode_ms` covers query embedding only; the offline memory embedding build cost is recorded in `memory_embedding_cache_manifest.json`.

## Example Aggregate Ranges

For `R=128`, `B=64`, across task/setting groups:

- `official_single`: p50 selection latency range 72.92-108.16 ms; p95 range 87.55-120.37 ms.
- `official_cross`: p50 selection latency range 92.79-138.30 ms; p95 range 129.92-181.40 ms.
- `causal_cross`: p50 selection latency range 91.87-140.10 ms; p95 range 130.20-178.23 ms.
