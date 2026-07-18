# Method 1 v30: Causal Graph Answerability Refinement

Status: `QWEN_INPUT_INVALIDITY_CONFIRMED`

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
