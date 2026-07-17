# Method 1 v21 QHDS Preregistration

This fixed-candidate-count phase uses the frozen Gate-B2-passing QESA r2 checkpoints and frozen Phase 1 UOT-FSA incidence. It does not alter the original Gate B negative result. The QHDS architecture and optimization settings are single-point preregistered; no QHDS hyperparameter is selected on an inner fold that contributed labels to the upstream frozen QESA model.

The learned utility is a nonnegative weighted sum of a concave aspect-coverage term and a modular direct-relevance term. Memory relevance is produced by a softplus network from query-memory cosine, QESA-incidence association, and their interaction. Aspect weights, incidence, relevance, and learned utility scales are nonnegative, and log1p is nondecreasing and concave. The resulting set utility is monotone submodular by construction.

Training uses only outer-train evidence groups. Each two-member group remains an OR set: its contrastive positive mass is the log-sum-exp over acceptable members, not a requirement to select both. Inference receives only query embedding, protected ANN seeds, eligible IDs, frozen incidence, memory embeddings, frozen QESA, and frozen QHDS parameters. Gold groups are used only for OOF evaluation.

The first phase reports B=8, 16, and 32 with five protected ANN seeds, including QHDS, QHDS without latent aspects, QHDS without UOT incidence, flat cosine, and QESA association. The final top-5 prefix is the same protected seed set for every method. Passing this phase does not establish a hypergraph contribution; pairwise factor-graph, fixed-byte, Tile, latency, oracle, and Pareto experiments remain mandatory before Gate C.

The quality equivalence interval is fixed at plus or minus 0.01 group coverage. Calibration and formal test remain locked and unread.
