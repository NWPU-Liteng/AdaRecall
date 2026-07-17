# Method 1 v21 Evidence-Bundle Locality r1 Phase Decision

The 5 seed x 5 session-blocked OOF experiment is complete for immutable attempt `attempt_006`. This phase used supervision-derived compositional evidence bundles, not observed runtime memory co-access.

The permutation-invariant learned scorer is retained as a negative result and is not approved as a sparse joint-hypergraph quality component. On K>=2 queries, it was worse than the independent-member comparator on all-groups-covered rate by -0.008092 (session-clustered 95% CI [-0.013258, -0.003466]) and on bundle recall by -0.023603 (CI [-0.033702, -0.013785]). It improved recall against pairwise-only by +0.016763 (CI [+0.007209, +0.025914]), but did not improve all-groups-covered rate (effect 0.0; CI [-0.004108, +0.004140]) or gold-compatible bundle hit.

The learned cardinality head was exact on 25.43% of multi-group OOF examples. Sparse candidate construction must therefore treat cardinality prediction as an unresolved bottleneck and must not silently expand a predicted singleton to manufacture pairwise edges.

The frozen source artifact contains 1,896 Cartesian-product rows, including 158 repeated-memory rows caused by overlapping alternatives across evidence groups. The frozen artifact remains unchanged. r1 fails closed by filtering any repeated-memory row before training; the per-outer-fold filtered count ranges from 113 to 137. This preserves the requirement that a usable bundle has one distinct memory for each evidence aspect.

Calibration and formal test were not read. This decision neither runs Gate C nor makes a PIM, physical Tile, or observed co-access claim.
