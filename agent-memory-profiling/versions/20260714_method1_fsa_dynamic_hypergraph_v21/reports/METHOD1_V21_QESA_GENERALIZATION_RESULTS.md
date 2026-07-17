# Method 1 v21 QESA Generalization Results

Gate B status: `not_passed`. All 25 fixed-seed, session-blocked outer OOF units were aggregated without replacing failed seeds or folds.

| Comparator (B=16) | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI | Holm p |
| --- | ---: | --- | ---: | --- | ---: |
| query_independent_aspect_prior | -0.003752 | [-0.007343, -0.000149] | -0.006599 | [-0.013624, +0.000325] | 0.0703965 |
| flat_cosine | -0.123715 | [-0.131233, -0.116275] | -0.113969 | [-0.126323, -0.101808] | 0.00019999 |
| shuffled_group_qesa | +0.005068 | [+0.000282, +0.009613] | +0.006551 | [-0.000668, +0.013656] | 0.0703965 |
| same_capacity_mlp_ranker | -0.050959 | [-0.057286, -0.044620] | -0.053324 | [-0.063888, -0.042453] | 0.00019999 |

Each two-member evidence group is an OR set of acceptable alternatives. K>=2 denotes the multi-group/compositional subset, not a higher-order member relation. Query IDs are metadata only and are absent from the QESA model interface.

Calibration and formal test remained locked and unread. Gate B passing permits QHDS development; it does not establish an independent hypergraph advantage over a matched pairwise factor graph.
