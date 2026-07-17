# Method 1 v21 Structure-Aware Selector OOF

All 25 units train a stateful selector on outer-train retained semantic-bundle provenance. The model has separate unary and structural marginal networks; structural state includes bundle completion/progress, prototype coverage novelty, and embedding diversity. Calibration and formal test were not read.

| selector | group recall | member recall | exact task success | payload Tiles | payload bytes | selector ms |
|---|---:|---:|---:|---:|---:|---:|
| same-frontier cosine | 0.685978 | 0.536475 | 0.613284 | 18.898 | 118545.5 | 0.0000 |
| existing QHDS | 0.687491 | 0.538553 | 0.613727 | 18.849 | 118689.5 | 0.0000 |
| structure-aware selector | 0.676052 | 0.523866 | 0.603690 | 18.736 | 118146.3 | 29.5158 |

No minimum-effect threshold is applied. This is an OOF selector revision; it does not read calibration or formal-test data and does not claim final Agent answer quality.
