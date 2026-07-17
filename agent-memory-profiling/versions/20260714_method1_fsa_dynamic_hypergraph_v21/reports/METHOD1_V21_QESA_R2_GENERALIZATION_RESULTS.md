# Method 1 v21 QESA r2 Generalization Results

Gate B status: `not_passed`. This is the single preregistered mechanism-driven revision after the immutable r1 negative result.

| Comparator (B=16) | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI | Holm p |
| --- | ---: | --- | ---: | --- | ---: |
| query_independent_aspect_prior | +0.007835 | [+0.003608, +0.012079] | +0.012765 | [+0.005270, +0.020376] | 0.000299985 |
| flat_cosine | -0.112128 | [-0.120090, -0.104130] | -0.094605 | [-0.107496, -0.081536] | 0.00019999 |
| shuffled_group_qesa | +0.007183 | [+0.002855, +0.011531] | +0.015414 | [+0.007407, +0.022989] | 0.00314984 |
| same_capacity_mlp_ranker | -0.039373 | [-0.045690, -0.032892] | -0.033960 | [-0.045415, -0.022183] | 0.00019999 |

Mean aspect-map pairwise cosine: `0.335003`. Mean query-conditioned prototype coordinate standard deviation: `0.007356`.

K>=2 is the multi-group/compositional subset; each inner group remains an OR set of two acceptable alternatives. Calibration and formal test remained locked and unread. If this revision does not pass, QESA and downstream QHDS promotion stop as a preserved negative result.
