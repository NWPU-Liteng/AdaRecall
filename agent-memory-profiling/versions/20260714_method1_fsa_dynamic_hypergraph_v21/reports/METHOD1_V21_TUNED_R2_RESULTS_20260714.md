# Method 1 v21 Post-Closure Tuning r2

This report records two new OOF-only revisions after the immutable core closure.
The original r2/r3 closure configs, attempt_019 artifacts, final freeze, calibration
read, and formal-test attempt are not modified.

## Protocol

- Data: frozen `1,355` train queries, five fixed seeds, five session-blocked outer folds (`25/25` units).
- Memory/index: `42,034` memories, frozen phase-1 UOT incidence with `64` prototypes, frozen sparse rank index, neutral `2,628` logical-Tile layout.
- Inference boundary: QESA prediction on GPU; sparse rank lookup, bundle construction, selector, payload gather, and Tile accounting on CPU. Model/index/incidence/payload are preloaded.
- Locked inputs: `calibration_read=false`, `formal_test_read=false` for both revisions.
- No access-order, co-access, LACA/OACH, Hawkes, PIM hardware, or fabricated dynamic stream is used.

## Changes

`tuned_r1` expanded the shared sparse frontier cap from `32` to `96` returned edge members. The same frontier is supplied to ANN flat, neural flat, QHDS hypergraph, and pairwise graph.

`tuned_r2` keeps that frontier and fixes selector work: QHDS and pairwise greedy recurrences stop at the maximum deployed budget `B=32`; they do not rank the unused frontier suffix.

## Pooled OOF comparison

At `activated_prototypes=16`, `members_per_semantic_edge=32`, `beam=32`:

| revision | B | group coverage | all groups covered | payload bytes | logical Tiles | query e2e |
|---|---:|---:|---:|---:|---:|---:|
| immutable r3 | 16 | 0.6094 | 0.5376 | 60,186 | 52.61 | 2.45 ms |
| tuned r2 | 16 | 0.6094 | 0.5373 | 60,267 | 52.61 | 5.01 ms |
| immutable r3 | 32 | 0.6587 | 0.5861 | 115,166 | 52.61 | 2.46 ms |
| tuned r2 | 32 | 0.6875 | 0.6137 | 118,689 | 52.61 | 5.01 ms |

The expanded frontier improves B=32 quality by `+0.0288` group coverage and
`+0.0276` all-groups-covered, but costs about `+3.5 KB` payload and does not
reduce logical Tile activations. The r2 cutoff reduces tuned-r1 selector mean
latency from `8.06 ms` to `2.56 ms` and mixed e2e mean from `14.01 ms` to
`8.56 ms` (p50 `11.27 -> 5.19 ms`, p95 `27.14 -> 21.82 ms`).

Against the tuned-r2 pairwise graph at the same `16/32/32, B=32` point, the
hypergraph effect is group coverage `+0.1175` (session-clustered CI
`[0.1055, 0.1247]`), logical Tile effect `0`, payload effect `+6,188 bytes`,
and query-e2e effect `-0.800 ms`.

## Interpretation

The tuned revision is a quality/latency Pareto improvement over the original
implementation at B=32, not a simultaneous win on every cost axis. Tile totals
remain metadata-dominated; they are host/logical metrics only. Real memory
increment/update metrics remain unavailable because the source data failed the
causal insertion-order audit, so the static FSA-QEHG fallback remains locked.

## Artifacts

- Config: `configs/v21_sparse_joint_tuned_oof_r2.json`
- OOF summary: `aggregate/v21_sparse_joint_core_summary_tuned_r2_attempt001.json`
- Fixed-count: `aggregate/v21_sparse_joint_fixed_count_tuned_r2_attempt001.csv`
- Latency summary: `aggregate/v21_sparse_joint_latency_summary_tuned_r2_attempt001.json`
- Raw metrics: `aggregate/v21_sparse_joint_core_metrics_tuned_r2_attempt001.csv`
- Raw latency: `aggregate/v21_sparse_joint_raw_latency_tuned_r2_attempt001.csv`

The OOF summary records SHA-256 hashes for all generated artifacts. No calibration
or formal-test rerun is authorized by this revision.
