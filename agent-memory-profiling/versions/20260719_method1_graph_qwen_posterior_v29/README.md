# Method 1 v29: Causal Graph Evidence Posterior

Status: `GRAPH_ONLY_QWEN_FAILED`

v28 passed Frontier Gate C but failed Qwen Gate D because Qwen promoted only
5-6 of the graph-novel gold memories into Top-5. v29 optimizes that failed
design point without changing the Method 1 candidate path.

Candidate IDs, graph provenance, causal snapshots, budgets, Qwen scores, model,
prompt, and safety results are immutable v28 inputs. v29 performs only a
query-local rank posterior between Qwen evidence likelihood and a causal graph
prior (`x_sem`, total submodular marginal, or selection order). The prior kind
and Qwen weight are selected on four folds and applied to the fifth fold.

Dense, IVF, Lexical, and Full controls receive the same cross-fitted
Qwen/retrieval-rank fusion interface. Graph methods never receive their Dense
rank, Dense score, lexical score, fallback, protected slots, or new candidates.

This remains MedMemoryBench train OOF development evidence, not an independent
fresh claim. v28's failed Gate D is immutable and is not retroactively changed.

The immutable cross-fit run completed for all 1,355 queries. Neither graph
posterior passed Gate D against the original Dense-HNSW-32 Qwen ranking (or
against the cross-fitted Dense control). Simple query-local rank fusion is
therefore rejected as the closure mechanism and must not be tuned further on
these OOF labels. See `reports/CROSSFIT_OOF_FAILURE.md` and
`artifacts/crossfit_oof_failure_receipt.json`.
