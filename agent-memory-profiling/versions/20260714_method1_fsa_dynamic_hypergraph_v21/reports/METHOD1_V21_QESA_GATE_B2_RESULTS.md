# Method 1 v21 QESA Gate B2 Representation Validity

Gate B2 status: `passed`. This audit used frozen QESA r2 checkpoints on unseen session-blocked OOF queries and did not train or select a new model.

| Metric | Comparator | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI |
| --- | --- | ---: | --- | ---: | --- |
| group_affinity_score | query_independent_aspect_prior | +0.008222 | [+0.007350, +0.009100] | +0.006863 | [+0.005896, +0.007888] |
| group_affinity_score | shuffled_group_qesa | +0.010522 | [+0.009680, +0.011375] | +0.008825 | [+0.007848, +0.009838] |
| target_cross_entropy | query_independent_aspect_prior | +0.143629 | [+0.131506, +0.155711] | +0.126519 | [+0.110655, +0.142673] |
| target_cross_entropy | shuffled_group_qesa | +0.127989 | [+0.116875, +0.139034] | +0.093717 | [+0.079343, +0.107601] |

Mean aspect-map cosine: `0.335003`; maximum unit mean cosine: `0.692831`; mean prototype coordinate std: `0.007356`.

Passing this representation gate permits QHDS development only. Direct flat/MLP retrieval and the matched pairwise factor graph remain mandatory end-to-end Gate C baselines. K>=2 denotes the multi-group/compositional subset. Calibration and formal test remained unread.
