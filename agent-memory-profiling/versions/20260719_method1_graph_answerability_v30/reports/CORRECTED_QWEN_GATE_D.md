# Corrected-Qwen Gate D

Run: `v30_corrected_qwen_oof_20260719`

Final decision: both `semantic_graph_only_32` and `dual_graph_only_32` pass the
pre-frozen Gate D against Dense-HNSW-32. This is strict train OOF development
evidence and does not authorize fresh evaluation.

The control changed no candidate IDs. All methods used the same local Qwen3
Reranker 0.6B, instruction, 2,048-token budget, and suffix-preserving balanced
head-tail packing. Candidate identity violations and generation suffix
violations were both zero; all 8,130 full rankings were persisted.

Semantic Graph-only achieved nDCG@10 0.037759 (delta +0.005948, 95% CI
[-0.002982, +0.014878]), Recall@5 0.039176 (delta +0.006335, CI [-0.003137,
+0.015744]), and group coverage@5 0.076138 (delta +0.016175, CI [+0.003381,
+0.028905]). Precision@5 delta was +0.002214 with CI [-0.000148, +0.004576].
Dual was nearly identical and also passed all four lower bounds.

Standard-IVF remained approximately tied with Dense. Lexical remained the
strong control (nDCG@10 0.415478; Recall@5 0.483825; group coverage@5
0.545203). Full-HNSW-128 reached nDCG@10 0.075423 and Recall@5 0.085363, but
its estimated Qwen time was 7.30 seconds versus 1.82 seconds for 32 candidates.

The corrected run required 3,220.86 wall-clock seconds on four RTX 3090 GPUs.
The validity fix therefore establishes Gate D quality but also raises a clear
system cost. Gates E-G and the Dense/IVF Pareto test remain mandatory before
any closure or fresh claim.
