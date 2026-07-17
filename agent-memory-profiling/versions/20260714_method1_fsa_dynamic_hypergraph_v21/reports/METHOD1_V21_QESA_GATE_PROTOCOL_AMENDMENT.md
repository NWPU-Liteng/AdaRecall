# Method 1 v21 QESA Gate Protocol Amendment

The original Gate B remains recorded as not passed. It required the QESA latent-aspect predictor, before QHDS set selection, to outperform direct flat cosine and a same-capacity memory ranker on top-B retrieval. That criterion conflated two module roles: QESA predicts latent evidence aspects, while QHDS is responsible for combining direct semantic relevance, aspect coverage, protected seeds, and resource constraints into a selected set.

This amendment does not retroactively pass the original Gate B. It creates Gate B2, frozen before any QHDS result is observed. Gate B2 evaluates whether the already frozen QESA r2 checkpoints predict heldout evidence-aspect targets better than a query-independent prior and shuffled-group supervision on unseen session-blocked OOF queries. Direct flat and MLP retrieval remain mandatory strong baselines at Gate C, where the complete selector is evaluated.

Gate B2 uses two primary paired metrics. Group-affinity score is the mean predicted probability mass aligned with every OR evidence group's frozen UOT-FSA prototype affinity. Target cross-entropy compares the predicted prototype distribution with the normalized set-of-sets target; its effect is reported as baseline loss minus QESA loss so positive values favor QESA. Prototype recall at 8 and the frozen linear probe are diagnostic.

For both primary metrics and both required comparators, the pooled all-query and K>=2 session-clustered 95% confidence interval lower bound must be strictly positive, and every fixed seed point effect must be positive. The r2 aspect maps must also satisfy the preregistered non-collapse thresholds. Query IDs remain metadata only. Calibration and formal test remain locked and unread.

Passing Gate B2 permits QHDS development but does not establish end-to-end retrieval superiority, compositional selection benefit, or an independent hypergraph contribution. Those claims require the fixed-budget, iso-quality, iso-cost, and matched pairwise-factor-graph experiments at Gate C.
