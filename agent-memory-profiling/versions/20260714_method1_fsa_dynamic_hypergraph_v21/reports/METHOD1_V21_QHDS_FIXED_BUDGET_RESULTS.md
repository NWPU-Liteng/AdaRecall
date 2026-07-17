# Method 1 v21 QHDS Fixed-Budget Results

This phase is complete but is not Gate C; pairwise factor-graph, byte, Tile, latency, oracle, and Pareto comparisons remain pending.

| B | Comparator | All effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI |
| ---: | --- | ---: | --- | ---: | --- |
| 8 | flat_cosine | +0.006568 | [+0.002943, +0.010163] | +0.008960 | [+0.003234, +0.015083] |
| 8 | qesa_association | +0.053444 | [+0.047988, +0.058910] | +0.048603 | [+0.039997, +0.056929] |
| 8 | qhds_without_latent_aspects | +0.000640 | [-0.000295, +0.001570] | +0.000771 | [-0.000588, +0.002193] |
| 8 | qhds_without_uot_soft_incidence | +0.006568 | [+0.002990, +0.010222] | +0.008960 | [+0.003225, +0.015004] |
| 16 | flat_cosine | +0.005092 | [+0.001766, +0.008762] | +0.011272 | [+0.004907, +0.017589] |
| 16 | qesa_association | +0.117220 | [+0.109971, +0.124927] | +0.105877 | [+0.092156, +0.119786] |
| 16 | qhds_without_latent_aspects | +0.000246 | [-0.000293, +0.000825] | +0.000385 | [-0.000576, +0.001365] |
| 16 | qhds_without_uot_soft_incidence | +0.005092 | [+0.001569, +0.008508] | +0.011272 | [+0.005052, +0.017626] |
| 32 | flat_cosine | +0.002497 | [-0.001110, +0.006026] | +0.004576 | [-0.001265, +0.010487] |
| 32 | qesa_association | +0.154736 | [+0.144241, +0.165322] | +0.158574 | [+0.141365, +0.176261] |
| 32 | qhds_without_latent_aspects | +0.000098 | [-0.000342, +0.000542] | +0.000385 | [+0.000000, +0.000970] |
| 32 | qhds_without_uot_soft_incidence | +0.002497 | [-0.000976, +0.006085] | +0.004576 | [-0.001360, +0.010699] |

Each evidence group is an OR set. K>=2 is the multi-group/compositional subset. All methods preserve the same protected top-5 and nested B policy. Calibration and formal test remained unread.
