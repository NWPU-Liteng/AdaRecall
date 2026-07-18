# Method 1 v25: Degree-Calibrated Residual Frontier

This version is a failure-driven continuation of the frozen v24 closure. It does not modify v24 and must not read or rerun the already revealed test split.

## Algorithm change

1. Dense Top-C is always materialized as the safety anchor.
2. Hypergraph evidence may replace only the low-confidence dense tail.
3. Broad edges are IDF/degree penalized, and an edge requires overlap with at least two seeds.
4. Replacement count is dynamic and bounded; each replacement must pass graph-gain and dense-drop margins.
5. The final frontier remains exactly C unique, causal, same-namespace candidates.

## Evaluation protocol

The development split is deterministically divided by query ID into tuning and validation partitions. Hyperparameters are selected on tuning only. Promotion requires non-negative candidate recall and evidence-group coverage on validation, plus at least one strict positive validation signal. A promoted configuration may then be checked with the frozen Qwen reranker on development. No v25 test run is provided by design.
