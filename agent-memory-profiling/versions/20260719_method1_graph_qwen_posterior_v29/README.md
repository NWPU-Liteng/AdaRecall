# Method 1 v29: Causal Graph Evidence Posterior

Status: `FROZEN_FOR_STRICT_CROSSFIT_OOF`

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
