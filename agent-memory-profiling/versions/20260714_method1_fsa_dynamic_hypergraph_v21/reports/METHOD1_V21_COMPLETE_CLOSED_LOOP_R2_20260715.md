# Method 1 v21 Complete Closed Loop r2

This revision adds a corrected logical-Tile model, a persona-by-prototype sharded hyper-index, matched integrated e2e controls, sparse-frontier selector refit, and candidate-quality headroom analysis. All results are train OOF or explicitly labelled diagnostics. Calibration and formal test were not read.

## Promoted Method 1 path

```text
offline memory -> frozen embedding -> UOT fuzzy incidence
              -> persona x prototype rank shards

query embedding
    |-> protected ANN top-5
    `-> QESA active prototypes -> query-persona rank shards -> bounded edge members
                              -> arity-3 semantic bundle beam
                              -> shared cap-96 frontier
                              -> existing QHDS B=32 (quality-first)
                                 or cosine B=32 (latency-first)
                              -> single-copy payload gather -> Agent handoff
```

## Core quality

| path | group recall@32 | member recall@32 | exact task success | payload |
|---|---:|---:|---:|---:|
| persona-sharded frontier + QHDS | 0.687491 | 0.538553 | 0.613727 | 118689.5 B |
| same frontier + cosine | 0.685978 | 0.536475 | 0.613284 | 118545.5 B |

Sharding changes no returned member or selection. The post-closure cap-96 frontier remains responsible for the larger quality gain over immutable r3; QHDS adds only a small quality-first effect over same-frontier cosine.

## Corrected Tile advantage

| metric | global rank index | persona x prototype shards | change |
|---|---:|---:|---:|
| rank entries visited | 50818.7 | 2239.0 | -95.59% |
| metadata bytes accessed | 203274.6 | 8956.2 | -95.59% |
| corrected total logical Tiles | 110.276 | 68.609 | -37.78% |
| standalone sparse lookup | 1.310 ms | 0.724 ms | -44.73% |

The old `52.61` mixed metric is retired and is not directly comparable with these corrected totals. The new metric separately accounts for actual 4 KiB rank-prefix blocks and capacity-16 memory feature/payload Tiles.

## Matched integrated e2e

The single-grid controls reuse identical model weights and produce identical selections. Global mean e2e is `29.496 ms`; persona-sharded mean e2e is `28.717 ms`, paired delta `-0.779 ms`. Bundle construction and CPU selection now dominate the online path.

## Negative selector refit and remaining quality headroom

The fixed sparse-frontier MLP refit is not promoted: B=32 group recall is `0.679852`, below ANN `0.685978` and QHDS `0.687491`. Nevertheless, the cap-96 frontier gold upper bound is `0.809434`, leaving `0.121943` selector headroom and `0.190566` candidate-discovery gap.

## What remains before a full method claim

- Persist real retained bundle membership/provenance so the selector can optimize bundle completion rather than four flat features.
- Integrate incremental append/delete publication into persona-prototype shards; the existing static overlay stress is global-index based.
- Run the selected B-memory contexts through the Agent and measure final answer accuracy, token cost, and total latency.
- Obtain a newly authorized independent heldout protocol; v21 formal test remains non-reportable and is not rerun.
- Use a dataset with genuine creation/read events for causal dynamic claims. Current data supports static content maintenance only.

No minimum-effect threshold is used. Host/logical Tile results are not PIM hardware results.
