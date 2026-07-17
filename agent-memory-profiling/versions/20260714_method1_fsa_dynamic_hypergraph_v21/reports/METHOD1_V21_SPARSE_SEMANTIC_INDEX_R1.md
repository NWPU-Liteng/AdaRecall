# Method 1 v21 Sparse Semantic Index r1

All 25 seed/fold UOT incidence matrices have immutable prototype-to-memory rank indexes. Index construction is offline only. Online lookup may inspect only rank-list prefixes until it finds the configured number of eligible members, and must report every visited incidence entry.

- Rank-index shape per unit: `[64, 42034]`
- Metadata bytes per unit: `10760704`
- Total immutable rank metadata bytes across 25 OOF units: `269017600`
- No calibration or formal-test input was read.

The retained learned bundle-locality scorer is not an approved primary component. It may only appear in a diagnostic ablation; the sparse joint primary candidate path starts from independent semantic-edge member ranking and must still face the budgeted pairwise factor graph.
