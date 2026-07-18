# Experiment Version Index

Updated: 2026-07-18

## Active Packages

The v21 package remains the consolidated historical mainline. The independent
v24 package is the PRISM-aligned pre-reranker closure implementation and keeps
its failed formal quality Gate as an immutable baseline. The separate
`motivation_experiments/` directory remains a read-only historical track.

| Package | Role |
|---|---|
| [`20260714_method1_fsa_dynamic_hypergraph_v21`](20260714_method1_fsa_dynamic_hypergraph_v21) | Method 1 FSA-QEHG mainline, all current v21 configurations/results, consolidated inputs, and historical provenance. |
| [`20260718_prism_aligned_method1_closure_v24`](20260718_prism_aligned_method1_closure_v24) | PRISM-aligned Qwen3 dual-view hypergraph closure. Engineering, development, test-once, and replay paths completed; formal Gate E quality failed and is preserved without test retuning. |

The independent motivation archive is documented at
[`motivation_experiments/README.md`](motivation_experiments/README.md). It
contains only motivation/workload-characterization packages and does not feed
the v21 training, calibration, or formal-test paths.

Start with [`CONSOLIDATION_20260717.md`](20260714_method1_fsa_dynamic_hypergraph_v21/CONSOLIDATION_20260717.md), then read the review packet at
[`reports/METHOD1_V21_EXPERIMENT_REVIEW_PACKET_20260717.md`](20260714_method1_fsa_dynamic_hypergraph_v21/reports/METHOD1_V21_EXPERIMENT_REVIEW_PACKET_20260717.md).

## Consolidated Inputs and Evidence

- `v21/inputs/` contains the only files read by the active v21 configs: the
  MedMemoryBench v3 memory corpus/embeddings, QA-basis query split and
  embeddings, frozen train retrieval, eligibility config, and parent stop
  records.
- `v21/provenance/v20_parent/` contains v20 and v20-r1 parent manifests,
  reports, source, and compact artifacts.
- `v21/provenance/legacy_versions/` contains selected reports, configs, source,
  figures, and compact aggregate summaries from the non-motivation Method 1
  lineage. Exact byte-identical files are kept once; `SELECTION_INDEX.tsv` and
  `DEDUP_INDEX.tsv` record the selection and removed duplicates.
- `motivation_experiments/` contains the separately archived motivation
  packages. Their original version names are retained as subdirectories for
  traceability, while duplicate files are still represented only once.
- `v21/aggregate/archive_original_manifests/` preserves pre-migration v21
  manifests. Canonical path-rewritten copies are in
  `v21/aggregate/migrated_manifests/`.
- `v21/aggregate/MIGRATION_MANIFEST_20260717.tsv` records source path, canonical
  path, byte count, SHA-256, and whether an item was archived or deduplicated.
- `v21/provenance/consolidation_tools/` contains only the one-time migration
  scripts; they are audit evidence, not active experiment entry points.

## Removed Version Directories

The following former Method 1 version directories were consolidated into v21
and removed after the canonical input and provenance hashes were verified:

`20260602_172552`, `20260603_154211`,
`20260710_method1_access_basis_v1`, `20260710_method1_end_to_end_pilot_v1`,
`20260711_method1_qa_basis_formal_v1`,
`20260713_method1_fca_dynamic_hypergraph_v20`,
`20260713_method1_fca_dynamic_hypergraph_v20_r1`, `v0.1`, `v0.2`,
`v0.3`, and `v0.4`.

The motivation packages formerly named
`20260616_phase_aware_real_agent_v3_formal`,
`20260621_real_trace_adaptive_phase_v4_formal`,
`20260622_memorycd_candidate_frontier_v5_1_formal`,
`20260622_memorycd_native_v5_formal`, `20260703_method1_motivation`,
`20260704_method1_motivation_medmemorybench_v3`,
`20260705_method1_motivation_medmemorybench_v4_completion`,
`20260715_method1_motivation_medmemorybench_v5_alignment`, and
`20260716_method1_motivation_medmemorybench_v5p1_controlled` were moved into
the separate `motivation_experiments/` archive instead of being mixed into v21.

The old names remain searchable in original historical manifest fields and in
the migration TSV; these strings are audit provenance, not active filesystem
dependencies.
