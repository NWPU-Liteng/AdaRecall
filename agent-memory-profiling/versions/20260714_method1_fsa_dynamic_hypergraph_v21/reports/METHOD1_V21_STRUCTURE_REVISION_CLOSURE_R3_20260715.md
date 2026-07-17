# Method 1 v21 Structure Revision Closure R3

## Retained provenance

The revision deterministically rebuilt the frozen cap-96 frontier and persisted true retained semantic-bundle membership and provenance for all 25 OOF units. All frontiers match the prior fixed frontier exactly.

| item | result |
|---|---:|
| OOF queries | 6,775 |
| retained semantic bundles | 216,800 |
| returned semantic edges | 108,400 |
| exact frontier equivalence | true |

## Structure-aware selector

The selector uses separate unary and stateful structural branches. Its greedy state includes bundle completion/progress, prototype coverage novelty, and embedding diversity. It was trained only on outer-train queries and evaluated on 25 target folds.

| B=32 selector | group recall | member recall | exact success | selector ms |
|---|---:|---:|---:|---:|
| existing QHDS | 0.687491 | 0.538553 | 0.613727 | 0.000 |
| structure-aware revision | 0.676052 | 0.523866 | 0.603690 | 29.516 |

This is a retained negative result. The structure-aware selector is not promoted, and existing QHDS remains the selector on the R2 deployment path. The next quality revision targets QESA/candidate discovery because the remaining cap-96 candidate gap is upstream of selection.

No minimum-effect threshold is used. Calibration and formal test were not read. Persona-sharded incremental publication is not integrated, Agent answer execution is not evaluated, and host/logical Tile results remain non-PIM measurements.
