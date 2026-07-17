# Method 1 v21 Matched Single-Grid E2E Control

Global and persona-sharded rank indices were evaluated with the same single grid, reused selector weights, and identical selections in all 25 OOF units. Calibration and formal test were not read.

| stage | global mean | persona-sharded mean | paired delta |
|---|---:|---:|---:|
| ann_seed_search_ms | 0.409 ms | 0.409 ms | -0.000 ms |
| latent_aspect_predict_ms | 0.920 ms | 0.918 ms | -0.003 ms |
| semantic_edge_lookup_ms | 5.042 ms | 5.024 ms | -0.018 ms |
| evidence_bundle_build_ms | 29.138 ms | 29.190 ms | +0.051 ms |
| bundle_hyperedge_score_ms | 0.032 ms | 0.032 ms | +0.000 ms |
| submodular_select_ms | 5.670 ms | 5.656 ms | -0.014 ms |
| candidate_union_ms | 1.324 ms | 1.329 ms | +0.005 ms |
| selection_total_ms | 5.672 ms | 5.658 ms | -0.014 ms |
| query_e2e_ms | 42.917 ms | 42.941 ms | +0.024 ms |

Corrected logical Tiles change from `110.276` to `68.609` (`-37.78%`). No minimum-effect threshold is applied.
