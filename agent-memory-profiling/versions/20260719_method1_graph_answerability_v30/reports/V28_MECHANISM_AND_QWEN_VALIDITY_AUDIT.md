# v28 Mechanism and Qwen Validity Audit

Scope: gold-aware post-hoc development diagnosis. This report cannot promote a
configuration or authorize fresh evaluation.

## Reproduction and candidate loss

All 148 causal snapshots were rebuilt from the frozen v28 source/config, and
all 1,355 Semantic Graph-only candidate lists reproduced exactly. The legal
active graph pool contained at least one gold memory for 508 queries, but the
Top-32 selector retained gold for only 93. For 435 queries, gold was present in
the graph pool and then removed by the selector.

Pool gold that was not selected had median `x_sem` rank 139 and median
`x_sem=0.00187`. Selected false positives had median `x_sem=0.01074`. The
unselected gold also came from broader, less cohesive active edges (median
minimum degree 73; maximum cohesion 0.830) than selected false positives
(degree 37; cohesion 0.898). The current relevance-dominant objective therefore
suppresses evidence in broad/uncertain but valid graph regions.

This supports a later multi-resolution child-edge and balanced coverage
refinement. It does not support candidate Dense/BM25 scores or further Qwen
rank-fusion tuning.

## Qwen input validity

The exact local Qwen tokenizer produced a median prompt length of 786 tokens
for the 43,360 Semantic candidates. 39,344 prompts (90.74%) exceeded v28's
`max_length=512`; the tokenizer used right truncation, which removed the
assistant generation suffix. The recorded final-token yes/no score therefore
did not occupy the intended answer position for truncated prompts.

All 55 graph-novel gold candidates exceeded 512 tokens (mean 1,032, P95 1,495,
maximum 1,579). Shared gold was truncated in 39/40 cases. This is a material
validity defect, not merely a latency/quality hyperparameter.

The required controlled repair is a suffix-preserving prompt packer with a
single pre-frozen token budget and identical policy for Full, Dense-HNSW, IVF,
Lexical, Semantic Graph-only, and Dual Graph-only. Candidate IDs, provenance,
Qwen model, instruction, and Gate D bounds remain unchanged.

## Current decision

v28 and v29 remain immutable failed runs; their negative records are not
deleted. Their Qwen evidence is additionally tagged as input-invalid for long
prompts. Fresh remains locked. A candidate-identical corrected-Qwen control
must run before any new graph selector experiment so the two effects are not
confounded.
