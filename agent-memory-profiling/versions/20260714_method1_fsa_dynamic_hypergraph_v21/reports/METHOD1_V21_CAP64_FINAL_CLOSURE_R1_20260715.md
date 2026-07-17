# Method 1 v21 cap64 Final Closure

## Decision

`cap=64` is the promoted candidate frontier cap. `cap=96` is unnecessary for the current quality target; `cap=32` is too small and `cap=48` has less headroom. The selected serving budget is `B=8` under the current 1800-word Agent context limit.

## Completed results

| metric | result |
|---|---:|
| group recall@32 | 0.687638 |
| exact task success | 0.614022 |
| payload | 118696.7 B |
| corrected logical Tiles, global -> persona shards | 110.276 -> 68.609 (-37.78%) |
| rank entries visited reduction | 95.59% |
| matched e2e, global -> persona shards | 29.496 -> 28.717 ms |
| matched rows | 6775 (25/25 units, selections identical) |
| Agent answer accuracy, B8 | 0.259041 |
| Agent total latency, B8 | 3313.1 ms |

## What is closed

Offline index/models, sparse retrieval, cap64 frontier, exact persona-by-prototype sharding, corrected host/logical-Tile accounting, matched integrated e2e control, static append/delete/atomic publish stress, and train-OOF Agent execution are complete.

## Remaining gaps

The main quality gap is candidate discovery and context packing, not a need for 96 candidates. The structure-aware selector was negative and QHDS remains the selector. Agent accuracy is only a train-OOF diagnostic; B8/B16/B32 are indistinguishable because the context cap truncates later memories. Persona-sharded incremental publication still needs serving-path integration and snapshot rebuild optimization. Formal heldout and causal dynamic claims are not reportable.

All outputs are OOF or explicitly labelled stress/diagnostic results; no calibration/formal split was read and no minimum-effect threshold was used. Tile numbers are host/logical accounting, not PIM hardware measurements.
