# Method 1 v21 Matched Single-Grid E2E Control

Global and persona-sharded rank indices were evaluated with the same single grid, reused selector weights, and identical selections in all 25 OOF units. Calibration and formal test were not read.

| stage | global mean | persona-sharded mean | paired delta |
|---|---:|---:|---:|
| ann_seed_search_ms | 0.437 ms | 0.426 ms | -0.011 ms |
| latent_aspect_predict_ms | 0.830 ms | 0.806 ms | -0.024 ms |
| semantic_edge_lookup_ms | 3.457 ms | 3.336 ms | -0.121 ms |
| evidence_bundle_build_ms | 16.981 ms | 16.461 ms | -0.520 ms |
| bundle_hyperedge_score_ms | 0.031 ms | 0.030 ms | -0.001 ms |
| submodular_select_ms | 6.436 ms | 6.357 ms | -0.079 ms |
| candidate_union_ms | 1.013 ms | 1.001 ms | -0.011 ms |
| selection_total_ms | 6.438 ms | 6.359 ms | -0.079 ms |
| query_e2e_ms | 29.496 ms | 28.717 ms | -0.779 ms |

Corrected logical Tiles change from `110.276` to `68.609` (`-37.78%`). No minimum-effect threshold is applied.
