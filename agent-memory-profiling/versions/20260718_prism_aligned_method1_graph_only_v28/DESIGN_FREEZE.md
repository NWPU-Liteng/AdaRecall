# v28 Design Freeze Protocol

The design is not formally frozen until `scripts/04_freeze_design.py` succeeds
from a clean, committed source state. Synthetic results may be used to repair
correctness bugs before that point. MedMemoryBench gold metrics may not be used
to tune the main configuration after the freeze receipt is created.

The registered primary methods are `semantic_graph_only_32` and
`dual_graph_only_32`. The dual history weight is allowed to be exactly zero
when the frozen-policy co-retrieval view has insufficient support; this is the
pre-registered semantic-only contraction described by the guide, not a
replacement retrieval path.

The primary selector is a non-negative monotone submodular objective combining
graph relevance, active fuzzy-edge coverage, and incidence-only facility
location. It uses deterministic CELF lazy greedy and stable memory-ID ties.

Formal advancement is one-way:

1. Gate B synthetic and causal correctness.
2. Gate C frontier OOF versus Dense-HNSW-32 and Standard-IVF-32.
3. Gate D Qwen only for Graph-only methods that pass Gate C.
4. Gates E-G layout/I/O, end-to-end, incremental, and replay.
5. Gate H one unread holdout only after all preceding gates pass.
