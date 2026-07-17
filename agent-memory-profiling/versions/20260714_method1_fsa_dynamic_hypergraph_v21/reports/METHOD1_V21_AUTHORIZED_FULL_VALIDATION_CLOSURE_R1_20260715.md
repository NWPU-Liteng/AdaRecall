# Method 1 v21 Authorized Full Validation Closure

## What was fully validated

All currently authorized validation surfaces are complete: 25-unit train-OOF retrieval for active16/24/32, 25-unit matched global/sharded e2e controls for active16/24, corrected Tile OOF, deterministic static new-memory partition publication at 1/5/10/20% growth, atomic snapshot serving, overlay/compaction equivalence, and `1355 x 3` baseline Agent executions plus B8/B32 handoff revisions.

The default remains `cap=64`, active16, B8. Active24 is an optional quality tier. Active32 is diagnostic because its latency cost is large.

## Agent result

Baseline B8/B16/B32, balanced packing B8, lexical handoff B8, structured evidence B8, and structured evidence B32 all produced answer accuracy `0.259041` under the registered train-OOF scorer. The handoff changes did not improve accuracy. This is a negative result, not a reason to claim that the final Agent effect is closed.

## Boundary

The retrieval/index/Tile/static-publication engineering loop is closed. A full scientific claim of end-to-end Agent effectiveness is not possible yet: the authorized Agent score is low, calibration/formal data are locked, and the source data do not identify causal dynamic behavior. No threshold was used to hide small effects, and no locked split was read.

Next work must change the answer-generation/evaluation protocol or model, then repeat the same full train-OOF and newly authorized heldout validation. Simply increasing candidate count has been tested and is not sufficient.
