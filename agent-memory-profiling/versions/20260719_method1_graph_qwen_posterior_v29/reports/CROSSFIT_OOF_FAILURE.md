# v29 Cross-fit OOF Failure

Final state: `GRAPH_ONLY_QWEN_FAILED`

The immutable run `v29_crossfit_oof_20260719` evaluated 1,355 queries using
strict five-fold cross-fitting. Each held-out fold used a Qwen weight and graph
prior selected only on the other four folds. Candidate IDs and full Qwen
scores were frozen v28 inputs. Candidate identity violations, Dense signals
for graph methods, and Dense fallback were all zero.

Neither Semantic nor Dual Graph-only passed Gate D against Dense-HNSW-32 plus
the original Qwen ranking. Semantic posterior nDCG@10 was 0.010104 (delta
-0.004559, 95% CI [-0.009229, -0.000193]); Recall@5 was 0.010086 (delta
-0.006888, CI [-0.013161, -0.000984]); group coverage@5 was 0.021894 (delta
-0.005289, CI [-0.014145, 0.003201]). Dual was weaker: nDCG@10 0.010090,
Recall@5 0.008979, and group coverage@5 0.019557.

Dense-HNSW selected Qwen-only (`alpha_qwen=1.0`) in every fold. Semantic and
Dual usually selected a substantial graph prior on the four training folds,
but the posterior generalized worse on held-out queries. This falsifies the
proposed query-local linear rank posterior as a closure mechanism. Further
alpha-grid tuning on these labels is prohibited.

The Lexical control remains materially stronger (nDCG@10 0.293940, Recall@5
0.357196, group coverage@5 0.418081) and must remain visible in all subsequent
comparisons. v29 does not authorize fresh evaluation or any full-closure claim.
