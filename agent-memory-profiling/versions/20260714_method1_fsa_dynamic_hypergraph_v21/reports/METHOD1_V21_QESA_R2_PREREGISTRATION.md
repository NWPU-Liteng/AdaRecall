# Method 1 v21 QESA r2 Preregistration

QESA r1 failed the unchanged Gate B on 25 outer OOF units and remains an immutable negative result. A train-only diagnostic found that its latent aspect-to-prototype maps had pairwise cosine near 0.97 and that query-conditioned prototype predictions had mean L1 distance 0.043 from their unit mean. Oracle A then showed positive frozen-incidence headroom over flat cosine at B=8, 16, and 32, including the multi-group/compositional subset.

QESA r2 is the only mechanism-driven corrective revision. It retains the same Phase 1 incidence, five seeds, five session-blocked folds, protected seeds, budgets, eligible sets, and baselines. It adds soft-target cross-entropy against train-only evidence-group prototype targets, aspect-map diversity and sparsity penalties, and a cross-Query aspect-usage penalty. Latent count and one of two preregistered anti-collapse regimes are selected only on inner train/validation data.

Gate B is unchanged: at B=16 QESA must have a strictly positive pooled session-clustered 95% CI versus the query-independent prior, flat cosine, shuffled-group QESA, and same-capacity MLP; every seed point effect must be positive; and every comparator must also have a strictly positive pooled K>=2 session-clustered CI lower bound.

If r2 does not pass, QESA is frozen as a negative result and QHDS is not promoted. No further QESA tuning revision is permitted. Calibration and formal test remain locked and unread.
