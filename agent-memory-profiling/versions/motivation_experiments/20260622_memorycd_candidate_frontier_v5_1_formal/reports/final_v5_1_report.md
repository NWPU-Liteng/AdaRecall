# Final V5.1 Report

This report is intentionally incomplete because full V5.1 formal task-quality curves and final full task runs have not yet been executed.

## Completed

- MemoryCD data moved to `data/public_datasets/memorycd/`.
- Dataset inventory generated.
- Full protocol manifests generated for `official_single`, `official_cross`, and `causal_cross`.
- Runtime manifests and evaluator labels are physically separated.
- Tenant isolation, causal time scope, and leakage audit passed on full manifests.
- FAISS flat retrieval was installed and validated against brute-force Top-R score-boundary tolerance during cached selection sweep.
- BGE-base and MiniLM-L6 local models were used in the cached selection sweep.
- Reranker batch size was `16`, not `1`.
- Full memory embedding cache was built for 314,116 memory rows.
- Cached selection sweep was run for the first 1000 eligible rating/ranking events per protocol, producing 33,000 trace rows.

## Not Yet Completed

- Full Section 7.2 selection sweep matrix over all eligible rating/ranking events.
- Qwen task-quality subset curve.
- Bootstrap confidence intervals.
- Pareto-min-budget selection.
- Full formal task runs.

## Current Answer To Guide Checklist

1. Tenant/protocol isolation: passed on full generated manifests.
2. Causal protocol future-memory audit: passed on full generated manifest.
3. Main path: one selection pass; no phase/controller path was used.
4. R/B/K: recorded in cached selection traces.
5. Timing vs B: cached 1000-event-per-protocol selection sweep exists; full 15,504-event curve not yet.
6. Task metrics vs B: not run yet.
7. High-budget reference / Pareto B*: not selected yet.
8. Evidence for budget compression: not evaluated yet.
9. Tasks not supporting rerank compression: not evaluated yet.
10. V6 comparison curve: not available until full V5.1 sweep and task curves are complete.
