# Fresh Holdout Availability Audit

This audit inspected only local dataset names and prior run/manifest names. It
did not open a new holdout gold file, did not create a read lock, and did not
consume a fresh attempt.

- LongMemEval is ineligible: full and LongMemEval-S formal/motivation runs are
  present from 2026-06-16, 2026-06-21, and 2026-07-03.
- LoCoMo is ineligible: it is explicitly consumed in the closure guide and has
  multiple prior formal runs.
- MSC is ineligible: formal reranking and replay runs are present from
  2026-06-16 and 2026-06-21.
- MemoryCD is ineligible: real-data candidate-frontier and Qwen task-quality
  runs are present from 2026-06-22.

No other local dataset was found with both a compatible causal
memory-query-evidence contract and unread gold provenance. The correct state is
therefore `METHOD1_GRAPH_ONLY_OOF_CLOSED_AWAITING_FRESH`, not full closure.
