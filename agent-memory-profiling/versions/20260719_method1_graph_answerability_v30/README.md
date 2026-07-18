# Method 1 v30: Causal Graph Answerability Refinement

Status: `METHOD1_GRAPH_ONLY_OOF_CLOSED_AWAITING_FRESH`

v30 continues the Graph-only Method 1 path after v28 passed Frontier Gate C
but failed Qwen Gate D and v29 rejected linear graph/Qwen rank fusion. It does
not add memory ANN, candidate query-memory cosine, Dense or lexical candidate
scores, protected slots, fallback, or Qwen feedback into candidate generation.

The first phase is a reproducible post-hoc diagnosis of the immutable v28
Frontier and full-ranking Qwen rows. It rebuilds every causal graph snapshot,
requires exact reproduction of Semantic Graph-only candidate IDs, and locates
loss at edge activation, pool selection, and Qwen ranking. Gold is used only
by this diagnostic evaluator. Its output is development evidence and cannot
promote a configuration.

Any later v30 algorithm must be frozen in a new commit before a new immutable
OOF run. It must refine the existing coarse/fuzzy/child-edge path rather than
replace it, and all 32 candidates must retain active-edge provenance.

The diagnostic reproduced all 1,355 Semantic candidate sets exactly. Its
active graph pool contained gold for 508 queries, while the selector retained
gold for only 93; 435 queries lost gold between the legal graph pool and
Top-32. Separately, exact tokenizer auditing found that 90.74% of Semantic
candidate prompts exceeded v28's 512-token budget. Right truncation removed
the assistant generation suffix, so the final-token yes/no score was invalid
for those prompts. All 55 graph-novel gold candidates were affected.

The first controlled repair will keep every v28 candidate ID fixed and apply
one suffix-preserving Qwen packing policy to all retrieval methods. This must
be frozen and evaluated before changing the graph selector, so interface and
candidate effects remain identifiable.

The immutable candidate-identical corrected-Qwen OOF run passed Gate D for
both Semantic and Dual Graph-only. Candidate identity and generation suffix
violations were zero. Semantic improved mean nDCG@10, Recall@5, and group
coverage@5 relative to Dense-HNSW-32; every pre-frozen lower CI bound remained
above -0.01. This is train OOF development evidence, not fresh confirmation.

Fresh remains locked until physical Tile/I/O, end-to-end Pareto, and causal
incremental Gates E-G are completed. The 2,048-token validity repair also
increased estimated 32-candidate Qwen time to about 1.82 seconds, so quality
closure does not imply system closure.

The immutable physical Tile and end-to-end run passed Gate E for Semantic and
Dual layouts and Gate F for both graph methods. Semantic layout reduced mean
activated Tiles by 73.36%, fetched bytes by 73.53%, and requested-cold read
latency by 70.30% versus chronological layout. Co-retrieval layout failed and
is retained as a negative Access result. Graph methods were slower than
Dense/IVF by 16-34 ms but had higher final quality, so neither baseline
strictly dominated them; both remained much faster than Full-128.

Gate G passed on 30,755 causal insertions, 148 exact query-boundary
compactions, and 1,355 explicitly named frozen-policy co-retrieval events.
Every external snapshot hash, candidate identity, layout check, and causal
safety check matched with zero violations. Delta insertion P50/P95 was
0.285/0.407 ms and mean write amplification was 1.021x. Compaction is
accurately reported as a full exact boundary rebuild (mean 1.343 s), not as a
local incremental compaction claim.

Gates C-G are now closed on strict train OOF development evidence. LongMemEval,
LoCoMo, MSC, and MemoryCD all have prior gold-bearing runs in this workspace,
so none is eligible as a never-read fresh holdout. Fresh remains locked until a
new compatible dataset can be frozen under the one-read protocol.
