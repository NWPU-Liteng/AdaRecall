# Runtime And Traffic Report

This is a cached selection-sweep report over the first 1000 eligible rating/ranking events per protocol, not the full V5.1 formal sweep and not task-quality evaluation.

The current selection traces include:

- `N`, `R_requested`, `R_effective`, `B_requested`, `B_effective`, `K`, `K_effective`
- dense seed IDs, frontier IDs, reranked IDs, final context IDs
- `t_scope_ms`, `t_query_encode_ms`, `t_seed_ms`, `t_frontier_ms`, `t_payload_fetch_ms`, `t_rerank_ms`, `t_context_pack_ms`
- `header_bytes_scanned`, `header_bytes_returned`
- `payload_bytes_for_rerank`, `payload_bytes_for_qwen_context`
- `rerank_pair_count`, `rerank_pair_token_sum`, `reranker_pair_batch_size`

Memory embeddings are now precomputed in:

```text
manifests/memorycd_official_single/embedding_cache/
```

The three protocol memory banks have identical SHA256 hashes, so this cache is reused for `official_single`, `official_cross`, and `causal_cross`. `t_query_encode_ms` in the current cached traces contains query embedding only; memory embedding build cost is recorded separately in `memory_embedding_cache_manifest.json`.

Current cached run:

```text
protocols: official_single, official_cross, causal_cross
tasks: rating_prediction, item_ranking
events per protocol: 1000
R: 32, 64, 128
B: 8, 16, 32, 64 where B <= R
rows per protocol: 11000
total rows: 33000
summary rows: 264
retrieval audit: faiss_flat Top-R score-boundary tolerance 1e-6
reuse policy: query_encoded_once_per_event; rerank_reused_per_B
```

Summary CSV:

```text
reports/candidate_frontier_sweep_summary.csv
```
