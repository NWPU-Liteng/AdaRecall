# V3 Reuse Manifest

Source v3 package:

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3
```

V4 package:

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260705_method1_motivation_medmemorybench_v4_completion
```

## Reused by Symlink

The following v3 artifacts are reused unchanged by symlink:

- `aggregate/M1_history_scale_invocations.jsonl`
- `aggregate/M1_history_scale_summary.csv`
- `aggregate/M2_frontier_budget_invocations.jsonl`
- `aggregate/M2_frontier_budget_summary.csv`
- `aggregate/M2_frontier_occupancy.csv`
- `aggregate/M3_tile_scattering.csv`
- `aggregate/O1_formal_stage_trace.jsonl`
- `aggregate/O1_same_cross_stage_stats.csv`
- `aggregate/O1_stage_shuffle_null.csv`
- `aggregate/O1_pair_recurrence.csv`
- `aggregate/medmemorybench_schema_audit.csv`
- `aggregate/medmemorybench_schema_stats.json`
- `aggregate/memory_items_turn_pair.jsonl`
- `aggregate/queries_medmemorybench_en.jsonl`
- `aggregate/query_evidence_mapping.jsonl`
- `aggregate/query_evidence_mapping_coverage.csv`
- `aggregate/embedding_cache/`
- `datasets/DATASET_PROVENANCE.md`
- all files under `configs/`
- all files under `schemas/`

## Copied for Local Modification

The following scripts were copied into v4 so they can be modified without touching v3:

- `scripts/medmmb_common.py`
- `scripts/run_o1_stage_trace.py`
- `scripts/compute_o1_null_models.py`
- `scripts/monitor_medmmb_v4_progress.py`

## New V4 Outputs

All new v4 scripts, manifests, aggregates, figures, and reports are written under the v4 package only.
