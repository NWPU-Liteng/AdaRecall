# Candidate-identical Corrected-Qwen Control Design

The immutable v28 Frontier candidate IDs are the only retrieval input. No
candidate, provenance record, score, or order may be changed before Qwen.

The control fixes a validity defect in v28's prompt truncation. Every prompt is
split at the document boundary into a chat prefix, document tokens, and the
assistant generation suffix. Prefix and suffix are always preserved. The
frozen total budget is 2,048 tokens, selected from the tokenizer-only audit to
cover 99.79% of all Semantic prompts, not from quality results. Documents that
still exceed the budget use a fixed 50/50 head-tail token packing policy.

Full-128, Dense-HNSW-32, Standard-IVF-32, Lexical-32, Semantic Graph-only-32,
and Dual Graph-only-32 receive the identical model, instruction, packing, and
yes/no scoring implementation. Candidate-level Qwen scores do not flow back
into the graph or selector.

Gate D bounds, metrics, bootstrap count, query folds, and Dense reference are
unchanged. This is a controlled interface-repair experiment, not a new graph
candidate result. It remains consumed train OOF development evidence and does
not authorize fresh evaluation.
