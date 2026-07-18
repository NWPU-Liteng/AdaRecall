# v29 Design Freeze

Frozen input lineage:

- v28 frontier rows SHA256: `834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98`
- v28 full-ranking Qwen rows SHA256: `439d32655eed0d6e136a181c9b66d373ad54e13937fbe16fe8131eea9e559696`
- v28 graph config SHA256: `d6252a2e669cb07226393d2bd6b90888f8973af8038dfaa4281b2b9174fb5e8b`

The registered alpha grid is `[0.25, 0.5, 0.75, 0.9, 1.0]`, where alpha is
the Qwen rank weight. Graph prior choices are `x_sem`,
`submodular_total_marginal`, and `selection_order`. Baseline controls use their
native frozen retrieval order. Ties prefer more Qwen weight, then the stable
prior name.

The inner objective is the unweighted sum of nDCG@10, Recall@5, and group
coverage@5, plus 0.25 times Precision@5. It is fixed before executing the
cross-fit script. No query-type-specific parameters are allowed.

Gate D remains the v28 non-inferiority bounds against the original
Dense-HNSW-32 + Qwen ranking. A stricter secondary comparison against the
cross-fitted Dense posterior is mandatory.
