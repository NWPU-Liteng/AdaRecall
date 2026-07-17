# Method 1 v21 Final Design Closure r2

## Final design decision

The frontier cap is fixed at `64`. `cap=32` is too small, while `cap=96` adds headroom without improving the selected result. The default serving tier uses `16` activated prototypes and `B=8`; a quality tier uses `24` activated prototypes and `B=8`. `32` activated prototypes is retained as a diagnostic point, not the default.

| tier | active prototypes | group recall@32 | exact success | core e2e | status |
|---|---:|---:|---:|---:|---|
| default | 16 | 0.687638 | 0.614022 | 28.0 ms | promoted |
| quality | 24 | 0.690836 | 0.617712 | 43.0 ms | promoted as optional |
| diagnostic | 32 | 0.694588 | 0.620959 | 79.4 ms | not promoted |

## Persona sharding and matched e2e

For the default B8 tier, corrected logical Tiles change from `110.276` to `68.609` (-37.78%); rank entries fall by `95.59%`. Matched e2e is `29.496 -> 28.717 ms`, with identical selections in all `6775` paired rows.

The active24 quality tier has corrected Tiles `140.929 -> 77.290` (-45.16%) and matched e2e `42.917 -> 42.941 ms`. Tile values are host/logical accounting, not PIM hardware measurements.

## Agent handoff conclusion

The original, balanced-packing, and lexical-order B8 evaluations all remain at `0.259041` train-OOF answer accuracy. Balanced packing increased generation latency without improving accuracy; lexical ordering also did not improve it. These are negative results and are not promoted. The remaining Agent gap is therefore answer generation/evidence adequacy, not simply candidate count or context order.

## Closed and bounded

The static retrieval path, cap64 frontier, persona-by-prototype sharding, corrected Tile accounting, matched e2e controls, persona-sharded atomic overlay dispatch, append/delete/compaction stress, and Agent train-OOF execution are implemented and validated. Structure-aware selector training was negative, so existing QHDS remains selected. Formal heldout performance and causal dynamic access/update claims remain outside the data/protocol boundary.

All inputs are OOF or explicitly labelled stress/diagnostic artifacts; calibration/formal splits were not read and no minimum-effect threshold was used.
