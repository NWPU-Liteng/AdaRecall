# Method 1 v21 Sparse-Frontier Selector Refit OOF

All 25 OOF units train the fixed candidate utility MLP on actual cap-96 outer-train sparse frontiers. Target folds, calibration, and formal test are excluded.

## B=32

| selector | group recall | member recall | exact task success | payload Tiles | payload bytes |
|---|---:|---:|---:|---:|---:|
| same-frontier cosine | 0.685978 | 0.536475 | 0.613284 | 18.898 | 118545.5 |
| existing QHDS | 0.687491 | 0.538553 | 0.613727 | 18.849 | 118689.5 |
| sparse-frontier refit MLP | 0.679852 | 0.529259 | 0.606494 | 18.847 | 118175.0 |

No minimum-effect threshold is applied. The result is retained whether the refit effect is positive, zero, or negative.
