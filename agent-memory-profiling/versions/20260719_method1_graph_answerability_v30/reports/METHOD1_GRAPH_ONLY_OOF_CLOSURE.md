# Method 1 Graph-only OOF Closure

Status: `METHOD1_GRAPH_ONLY_OOF_CLOSED_AWAITING_FRESH`

Method 1 now closes Gates C-G on strict five-fold train OOF development
evidence while preserving the Graph-only design. Query vectors align only to
causal edge descriptors; all 32 visible slots are graph-owned, and no
memory-level ANN, candidate query-memory cosine, Dense score, Dense tie-break,
protected slot, or Dense fallback enters the main method.

## Confirmed Results

- Gate C: Semantic and Dual passed Frontier. Relative to Dense-HNSW-32,
  Recall@32 improved by +0.007811 and +0.008180 respectively, with 32/32
  graph-owned slots and zero underflow or safety violations.
- Gate D: after repairing the invalid right-truncated Qwen prompt, Semantic
  improved over Dense on nDCG@10 (+0.005948), Recall@5 (+0.006335), and
  GroupCov@5 (+0.016175). The GroupCov gain has a positive 95% CI
  [+0.003381, +0.028905]; the other two positive means remain statistically
  uncertain because their intervals cross zero.
- Gate E: Semantic layout reduced mean activated Tiles by 73.36%, fetched
  bytes by 73.53%, and requested-cold mean read latency by 70.30% versus the
  chronological layout. Dual passed but was weaker. Co-retrieval failed and is
  retained as a negative Access result.
- Gate F: Semantic and Dual form a quality-system Pareto point. Semantic total
  warm latency was 1,901.16 ms, about 33.49 ms slower than Dense but 5,441.12
  ms faster than Full-128; its final quality was higher than Dense and IVF, so
  neither strictly dominated it.
- Gate G: 30,755 physical delta insertions and 148 exact boundary rebuilds
  passed with zero causal or identity violations. Insertion P50/P95 was
  0.285/0.407 ms and mean write amplification was 1.021x. Boundary compaction
  remains a full exact rebuild averaging 1.343 s and is not claimed as local.

## Negative And Limiting Evidence

The original v28 Qwen run and v29 cross-fitted rank posterior failed Gate D;
simple graph/Qwen rank fusion is rejected. Mechanism diagnosis localized most
Frontier loss to broad, low-cohesion edges and motivated child-edge coverage.
The corrected prompt then showed that the earlier Qwen negative result was
also confounded by truncating the assistant scoring suffix.

Lexical BM25 remains substantially stronger: nDCG@10 0.415478, Recall@5
0.483825, and GroupCov@5 0.545203. It is retained as a control and prevents a
claim that Graph-only is the best quality retriever. The supported conclusion
is narrower: Graph-only beats Dense/IVF means at the same 32-candidate budget
and provides strong semantic physical locality.

## Fresh Boundary

No new gold was read. LongMemEval, LoCoMo, MSC, and MemoryCD all have prior
gold-bearing runs in this workspace, so none qualifies as a never-read fresh
holdout. Gate H remains locked (`fresh_frozen.yaml` is `NOT_FROZEN`), and full
independent closure cannot be claimed until a compatible new dataset is
frozen and consumed exactly once under the guide protocol.
