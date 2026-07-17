# LIMITATIONS

- This package consolidates historical formal runs instead of re-running a new guide-perfect M1 matrix.
- Historical V3 cross-encoder raws do not preserve `gpu_name` and `git_commit` per record; the suite fills `gpu_name=unknown_historical_v3` and `git_commit=unavailable` rather than inventing metadata.
- The available historical frontier budgets are `{50, 100}` rather than the guide-preferred `{8, 16, 32, 64}`.
- M2 uses a task-proxy based on synthetic query gold-group hit and high-budget stability because the historical V3 path is not an official QA benchmark run.
- The fresh V4 official-QA budget grid was fully rerun at `{8, 16, 32, 64}` with a metadata-aware answer path and corrected scorer, but the formal task scores remain low and non-monotonic: `locomo` stays in the `1.6%-4.7%` EM band and `longmemeval_s` stays in the `12.5%-15.6%` EM band across budgets.
- The final QA path is transparent but no longer "raw generation only": it uses deterministic short-answer contraction plus scorer-side normalization for explicit acceptable alternatives and long-form insufficiency references. This improves reliability, but it should be described as part of the experiment pipeline rather than mistaken for an untouched benchmark script.
- The fresh query strata are auditable but partially heuristic: `single_evidence` / `multi_evidence` use native labels when available and otherwise fall back to `gold_turn_ids` cardinality, while `long_range` is defined by the oldest-quartile gold-memory position within each dataset's visible history rather than a benchmark-native long-range tag.
- The `longmemeval_s` formal runs expose an additional controller-side limitation: many hard-subset queries do not fill the configured `final_top_k=5`, so high-budget stability can remain well below `1.0` even at budget 64 simply because the selected frontier is sparse or empty.
- The `locomo` formal runs expose a different limitation: even after answer-path cleanup, many misses are still driven by incomplete evidence capture rather than answer formatting, so task quality remains retrieval-limited there.
- O1 relies on the real-trace V4 `selection_only` runs, which preserve real retrieval decisions but omit answer generation and narrow the available phases to `planning` and `reasoning`.
- The workspace `.git/` directory is empty, so historical commit reconstruction is not possible from local metadata.
