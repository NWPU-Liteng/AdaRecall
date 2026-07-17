# Method 1 v21 Consolidated Experiment Package

This directory is the single canonical package for the Method 1 experiment.
The previous date/version directories under `versions/` were consolidated here
on 2026-07-17. The active implementation and all current v21 results remain in
their original `aggregate/`, `artifacts/`, `configs/`, `reports/`, `scripts/`,
`src/`, and `tests/` locations.

## Canonical layout

- `inputs/`: immutable files that v21 reads during development and evaluation.
  The files are copied from the frozen v3 memory corpus and QA-basis v1, with
  stable v21-local names.
- `provenance/v20_parent/`: v20 and v20-r1 manifests, freeze records, source
  code, reports, and compact result summaries needed to audit the parent chain.
- `provenance/legacy_versions/`: reports, configurations, source code, figures,
  and compact aggregate summaries from the non-motivation Method 1 lineage.
  Large raw traces, repeated shards, caches, and checkpoints are intentionally
  omitted; the selection and hash mappings are recorded in `SELECTION_INDEX.tsv`
  and `MIGRATION_MANIFEST_20260717.tsv`. The selection index is global and also
  records files moved to the separate motivation archive.
- `../motivation_experiments/`: a separate read-only archive for the earlier
  real-agent, MemoryCD, and MedMemoryBench motivation packages. These packages
  support workload/motivation claims but are not active v21 inputs or model
  variants. See its `README.md` for the inventory and reading order.
- `provenance/consolidation_tools/`: one-time migration scripts retained as
  provenance; they are not active v21 commands and must not be rerun.
- `provenance/historical_diagnostics/`: diagnostic code whose retained outputs
  are useful for audit but whose original all-train input was intentionally not
  retained; it is not an active rerun path.
- `aggregate/archive_original_manifests/`: pre-migration copies of v21 manifests
  and configs whose historical paths were rewritten in the active configuration.

## Active input mapping

| v21 consumer | Canonical path |
|---|---|
| Memory items | `inputs/memory_medmemorybench_v3/memory_items_turn_pair.jsonl` |
| Memory embeddings | `inputs/memory_medmemorybench_v3/embedding_cache/embeddings.npy` and `ids.json` |
| Train/calibration/test query plans | `inputs/qa_basis_v1/query_split/query_plan_*.jsonl` |
| Query vectors/index | `inputs/qa_basis_v1/query_embeddings/query_vectors.npy` and `query_index.json` |
| Frozen train retrieval | `inputs/qa_basis_v1/frozen_train/retrieved_contexts_flat_graph_hyper.jsonl` |
| Eligibility source config | `inputs/qa_basis_v1/config/keypoint_basis_v6_schedule_layout_candidate.yaml` |

## Retention rule

The v21 package is runnable from this directory without sibling version
directories. Historical material is retained only when it provides a report,
method/configuration definition, figure, compact summary, frozen parent
evidence, or a unique negative result. Files that were exact byte-for-byte
duplicates are represented once; the removed source paths are listed in the
deduplication section of the migration manifest.

Historical manifests may still contain the original absolute path as an
evidence field. They are preserved under `aggregate/archive_original_manifests`
and are not used as active inputs.
