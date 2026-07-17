# Phase-Aware Real-Agent V3 README

## 1. This Version Is For

This version is the formal `v3` experiment package for the paper direction:

- keep the agent execution path intact instead of collapsing everything into a single offline retrieval call
- preserve phase-specific memory access behavior
- expose whether the optimization target is coarse retrieval, candidate processing, or rerank
- generate traces that can later be used to motivate hypergraph / multi-view memory layout design and PIM-oriented primitives

This README is intentionally implementation-facing. The goal is to make the current experiment non-black-box.

## 2. Version Layout

Current formal assets are stored under:

- [runs](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/runs)
- [configs_snapshot](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot)
- [phase_aware_real_agent_v3_experiment_report_20260616.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/phase_aware_real_agent_v3_experiment_report_20260616.md)

Active source code used by this version:

- [profile_longgraph_memory_selection.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)
- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)
- [reranker_client.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [embedding_client.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/embedding_client.py)
- [vector_store.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/vector_store.py)
- [import_public_dataset.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/import_public_dataset.py)
- [codex_phase_aware_handoff_20260615.md](/data2/user25263093/userliteng/26_code/preTest/codex_phase_aware_handoff_20260615.md)

## 3. What The Experiment Actually Measures

The experiment measures the cost of memory selection inside a real three-stage agent execution path.

The agent-side stages are:

1. `planning`
2. `reasoning`
3. `output`

The memory-side stages inside each selection pass are:

1. query encoding
2. coarse retrieval
3. metadata filtering
4. relation expansion
5. candidate aggregation / dedup
6. rerank
7. context construction

The core design choice is that this `v3` experiment uses:

- `agent_memory_mode = phase_aware`
- `agent_execution_backend = llm`
- `generation_backend = local_qwen`

That means the three agent phases are not mocked away. Each phase gets its own memory query and its own selection pass, then the selected memory for that phase is fed into the real LLM-based agent executor.

## 4. Why `phase_aware` Exists

The paper motivation is not just "memory retrieval is expensive". The stronger claim is:

- different agent phases read memory with different locality patterns
- these different locality patterns should eventually map to different logical / physical memory views
- PIM optimization should target the actual hot operators under those phase-specific access paths

So `phase_aware` is the minimal experiment setting that preserves this structure.

The implementation is in [_phase_plan(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py) and [_phase_query_text(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py).

The phase query suffixes are:

- `planning`: "Focus on plan setup, constraints, and long-term continuity."
- `reasoning`: "Focus on linked evidence, recent outcomes, and causal support."
- `output`: "Focus on concise answer support and user-facing grounding."

This is not yet a learned phase controller. It is a controlled prompt-level way to force the same base task to issue phase-specific memory requests, so the resulting trace can reveal locality differences.

## 5. Shared-Once vs Phase-Aware

There are two memory invocation policies in the code:

- `shared_once`
- `phase_aware`

`shared_once` means:

- run one selection pass
- reuse the same memory context across later agent stages

`phase_aware` means:

- planning issues a dedicated selection pass
- reasoning issues a dedicated selection pass
- output issues a dedicated selection pass
- a row is emitted into `phase_trace.jsonl` for each phase

So yes: in the current implementation, each phase really does re-run retrieval and rerank once.

This is deliberate, because the purpose here is to preserve phase-specific access behavior. It is not claiming that all production agents must always do this. It is claiming that if your future architecture wants to optimize phase-local memory access, you first need an experimental setup that actually exposes those phase-local accesses.

## 6. Overall Execution Pipeline

For each query, the profiler executes:

1. load a public-dataset-derived memory collection
2. build or load the ANN vector index
3. construct a query workload
4. run a first selection pass
5. if `agent_memory_mode = phase_aware`, re-run phase-specific selection for planning / reasoning / output
6. feed the phase-specific contexts into the real LLM-based agent executor
7. log one `raw_profile` row per query-setting pair and one `phase_trace` row per phase

The main control path is [_run_single_query(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py).

## 7. Retrieval Backbone

### 7.1 Embedding model

All formal `v3` runs use:

- `embedding_backend = transformer`
- model dir: [models/bge-base-en-v1.5](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/bge-base-en-v1.5)

The actual implementation is [TransformerEmbeddingClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/embedding_client.py).

What it does:

- tokenize text with HuggingFace tokenizer
- run `AutoModel`
- mean-pool token embeddings using attention mask
- L2-normalize the pooled vector

This produces a 768-dimensional embedding because the config sets:

- `embedding_dim = 768`

This is a dual-encoder retrieval setup: query and memory are embedded independently, then matched by vector similarity.

### 7.2 Coarse retrieval

All formal `v3` runs use:

- `retrieval_backend = ann_numpy`

Implementation:

- [AnnNumpyVectorStore](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/vector_store.py)

What it does:

- store normalized vectors
- build a lightweight centroid-based ANN index
- probe top clusters for a query
- score only vectors inside those probe clusters
- fallback to full scan if candidate coverage is too small

This is not a FAISS or Qdrant benchmark. It is a lightweight in-repo ANN implementation used to expose retrieval-stage latency and candidate-set size under controlled settings.

## 8. Candidate Processing Before Rerank

After coarse retrieval, the selection pipeline applies:

1. metadata filter
2. relation expansion
3. aggregation / dedup

### 8.1 Metadata filter

Implementation:

- [_metadata_filter(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

It keeps candidates whose `task_tags`, `entity_tags`, or `memory_type` match the query intent.

This is query-type sensitive:

- `fact` prefers type match plus task/entity overlap
- `temporal` allows broader overlap
- `relation` emphasizes task/entity overlap and structural memory types
- `plan_continuity` emphasizes procedural/profile/episodic continuity
- `constraint` emphasizes constraint/profile memories

### 8.2 Relation expansion

Implementation:

- [_expand_relations(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

It traverses explicit memory links up to `relation_expansion_depth`.

The inherited score decays by:

```text
next_score = inherited_score * (0.92 ** (current_depth + 1))
```

This models a simple intuition: structurally adjacent memories are useful, but indirect neighbors are usually weaker than directly matched items.

### 8.3 Candidate aggregation

Implementation:

- [_aggregate_candidates(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

Filtered and expanded candidates are merged by `memory_id`, keeping the highest score per item, then sorted descending.

This stage matters because your paper is not only about retrieval vectors. It is also about graph- or hypergraph-like memory structure, which changes candidate fan-out before rerank.

## 9. What `heuristic` Rerank Really Is

### 9.1 Implementation

Implementation:

- [HeuristicRerankerClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [_heuristic_rerank(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)
- [_rerank_score(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

### 9.2 Formula

The heuristic rerank score is:

```text
score
= vector_score
+ 0.22 * importance
+ 0.16 * recency
+ 0.10 * task_overlap
+ 0.08 * entity_overlap
+ relation_bonus
```

Where:

- `vector_score` is the coarse retrieval similarity
- `importance` is the memory-side scalar metadata
- `recency` is the memory-side scalar metadata
- `task_overlap` is the count of overlapping task tags
- `entity_overlap` is the count of overlapping entity tags
- `relation_bonus` depends on whether the memory belongs to the same gold group and on query type

Extra bonuses:

- `+0.18` if `memory.gold_group_id` is in `query.gold_group_ids`
- `+0.12` more for `relation` queries
- `+0.08` more for `plan_continuity` queries
- `+0.12` more for `constraint` queries when the memory type is `constraint`

### 9.3 Interpretation

This is not a neural reranker. It is a structured linear scoring rule that approximates a cheap rerank stage using metadata and retrieval score.

Why keep it:

- it provides a low-cost rerank baseline
- it lets you separate "retrieval / filtering / expansion cost" from "heavy neural rerank cost"
- it helps show whether the hotspot is inherently rerank-heavy or whether retrieval-like stages are already dominant

## 10. What `cross_encoder` Really Is

### 10.1 Model used in this experiment

All formal `cross_encoder` runs use:

- model dir: [models/cross-encoder-ms-marco-MiniLM-L6-v2](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/cross-encoder-ms-marco-MiniLM-L6-v2)

The model card says it is:

- a cross-encoder trained on MS MARCO passage ranking
- HuggingFace sequence-classification format
- `BertForSequenceClassification`
- 6 transformer layers
- hidden size 384

The local `config.json` confirms:

- `architectures = ["BertForSequenceClassification"]`
- `num_hidden_layers = 6`
- `hidden_size = 384`
- `model_type = "bert"`

Reference:

- [models/cross-encoder-ms-marco-MiniLM-L6-v2/README.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/cross-encoder-ms-marco-MiniLM-L6-v2/README.md)

### 10.2 Implementation

Implementation:

- [CrossEncoderRerankerClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [_cross_encoder_rerank(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

What it does:

1. form `(query_text, memory_text)` pairs
2. tokenize the pair jointly
3. run `AutoModelForSequenceClassification`
4. extract the relevance logit
5. combine the model score with a small residual vector-score term

The final score is:

```text
combined_score = model_score + 0.05 * vector_score
```

### 10.3 Why this is fundamentally different from the embedding retriever

The embedding retriever is a dual encoder:

- query and passage are encoded independently
- matching happens later by vector similarity

The cross-encoder is a joint encoder:

- query and passage tokens are fed into the same transformer
- the model outputs a direct relevance score

This is usually more accurate, but much more expensive. That is exactly why it is useful here: it makes rerank cost visible enough to test whether rerank is the real hotspot under phase-aware agent execution.

### 10.4 Why `reranker_batch_size = 1` in formal cross-encoder runs

The formal configs set:

- `reranker_batch_size = 1`

for the cross-encoder runs.

This makes the rerank latency visible and closer to a conservative online-serving regime. It also avoids hiding rerank overhead behind large batch amortization.

This is not claiming that batch size 1 is universally optimal. It is a profiling-oriented choice that makes per-query rerank cost easier to interpret.

## 11. What `oracle_pruned_rerank` Means

### 11.1 Implementation

Implementation is inside [_run_selection_pass(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py).

The logic is:

1. run normal retrieval, filtering, expansion, and aggregation
2. split aggregated candidates into:
   - `oracle`: candidates whose `gold_group_id` matches the query's `gold_group_ids`
   - `spill`: all other candidates
3. concatenate `oracle + spill`
4. truncate to `oracle_prune_limit`
5. rerank only this truncated set

In the formal configs:

- `oracle_prune_limit = 50`

### 11.2 Why this exists

This mode is not a deployable method. It is an analysis tool.

It answers the question:

- if we had a much better pre-rerank pruning policy that preserved the right locality / group structure, how much rerank cost could we remove before touching the agent LLM?

This is useful for your paper because your future hypergraph / PIM design is supposed to do something structurally smarter than today's generic host-side rerank.

So `oracle_pruned_rerank` is a controlled upper-bound style baseline:

- not "final system behavior"
- but "how much value is there if we can prune candidates earlier and more intelligently"

### 11.3 Why `gold_group_id` is not arbitrary here

`gold_group_id` is attached during dataset conversion.

For each public dataset:

- `LoCoMo`: `gold_group_id = conversation_id`
- `MSC`: `gold_group_id = session_id` or session-like conversation grouping
- `LongMemEval`: `gold_group_id = sample_id`

See:

- [import_public_dataset.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/import_public_dataset.py)

So the oracle uses known conversation/session/sample grouping as a proxy for the "right memory cluster" for a query.

That is why it is meaningful for analysis, even though it is not available at deployment time.

## 12. How Real The Agent Path Is In This Version

The agent executor is in:

- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)

Important details:

- backend is `llm`, not calibrated mock
- planning, reasoning, and output each run their own LLM rounds according to config
- the tool stage is simplified to deterministic memory inspection, so there is no external IO noise
- phase-specific contexts are actually passed into the corresponding agent stage

For the formal `v3` configs:

- `agent_execution_steps = 1`
- `agent_tool_calls_per_step = 1`
- `agent_planning_llm_rounds = 1`
- `agent_reasoning_llm_rounds = 1`
- `agent_output_llm_rounds = 1`

So the agent is not a complex multi-step tool-use benchmark. It is a controlled three-stage LLM agent pipeline with preserved phase structure and real LLM latency.

## 13. Dataset Mapping In This Version

The formal `v3` runs use the following converted memory files:

| Dataset | Memory file | Actual memory rows |
|:--|:--|--:|
| LoCoMo | `data/motivation_memory_selection/locomo_memories_1000.jsonl` | 5882 |
| LongMemEval | `data/motivation_memory_selection/longmemeval_memories_1000.jsonl` | 23867 |
| MSC | `data/motivation_memory_selection/msc_memories_5000.jsonl` | 395934 |

Important note:

- the filenames include `1000` or `5000` because they were created from earlier conversion/export workflows
- the formal `v3` configs correctly set `memory_sizes` to the real row counts used by the run
- so the actual selection scale is the row count above, not the suffix in the filename

## 14. Query Workload Design

The workload uses five query types:

- `fact`
- `temporal`
- `relation`
- `plan_continuity`
- `constraint`

Each query type is associated with:

- preferred memory types
- task tags
- entity tags
- temporal window
- one or more `gold_group_id` targets

The query construction utilities are in:

- [synthetic_memory_generator.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/synthetic_memory_generator.py)

Even when real memory files are used, the profiler still constructs `QuerySpec` objects with:

- task/entity hints
- query type
- preferred memory types
- target group IDs

This gives the experiment a consistent profiling interface across datasets.

## 15. Baselines In This Formal V3 Package

The formal runs compare two baseline modes:

- `full_selection`
- `oracle_pruned_rerank`

### 15.1 `full_selection`

This is the normal path:

- retrieve
- filter
- expand
- aggregate
- rerank the resulting candidate set
- take top-`k`

This is the main "do everything normally" baseline.

### 15.2 `oracle_pruned_rerank`

This keeps the same early selection stages but shrinks the rerank input using oracle group membership before rerank.

This is the analysis baseline used to estimate how much latency is caused by a too-large rerank frontier.

## 16. How Time Is Computed

There are two memory-selection notions in the logs, and this is easy to confuse.

### 16.1 `selection_total_latency_ms`

This is only the first selection pass latency:

- one pass
- one query
- one context

Under `phase_aware`, this is not the full memory cost the agent actually pays.

### 16.2 `phase_selection_total_latency_ms`

This is the sum over all agent phases:

- planning selection
- reasoning selection
- output selection

So in the formal `phase_aware` runs, this is the meaningful total selection cost for the whole response.

Similarly:

- `phase_retrieval_total_latency_ms` = sum of retrieval-like pieces across phases
- `phase_rerank_total_latency_ms` = sum of rerank latency across phases

### 16.3 Total response time

The code defines:

```text
total_response_latency_ms
= phase_selection_total_latency_ms
+ agent_execution_total_latency_ms
```

So the time-share metrics in the report are computed against:

- full phase-aware selection cost
- plus full agent execution cost

This is why `phase_selection_to_total_response_ratio` is the correct number to use for the paper-facing "memory selection share of total response latency" claim.

## 17. Why The Hotspot Moves To Rerank In Cross-Encoder Runs

Because `phase_aware` repeats selection three times, any heavy rerank backend is amplified by the number of agent phases.

So the total cost is roughly:

```text
selection_total
~ 3 * (retrieval-like cost + rerank cost + packaging cost)
```

When rerank is cheap, as in `heuristic`, selection remains a small share on smaller datasets and a moderate share on the largest dataset.

When rerank is expensive, as in `cross_encoder`, the rerank term dominates each selection pass, and then the three passes together push total selection cost close to agent-LLM cost.

That is the main methodological difference from your earlier `shared_once` style setting.

## 18. Formal Run Matrix

This version contains six formal runs:

| Dataset | Rerank | Output dir |
|:--|:--|:--|
| LoCoMo | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615` |
| LoCoMo | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615` |
| LongMemEval | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616` |
| LongMemEval | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616` |
| MSC | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616` |
| MSC | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616` |

## 19. What The Current Results Mean

At a high level:

- `heuristic` runs tell you what happens when rerank is lightweight
- `cross_encoder` runs tell you what happens when rerank is realistic and expensive

The main paper-facing message from this `v3` package is:

- once you preserve phase-specific memory invocation, the full memory-selection path can become a first-order latency term
- in that setting, the dominant hotspot is usually rerank, not coarse retrieval
- candidate pruning before rerank matters much more than shaving only ANN lookup cost

This supports the argument that future memory-system optimization should target:

- phase-specific locality
- candidate-set shaping
- rerank-adjacent primitives
- layout designs that reduce host-side data movement before expensive scoring

## 20. Limitations

This version is strong enough for motivation, but it is not yet the final architecture paper result.

Current limitations:

- phase-specific query rewriting is prompt-template-based, not learned
- the agent tool stage is simplified to deterministic memory inspection
- `oracle_pruned_rerank` is an analysis baseline, not a deployable method
- the ANN retriever is an in-repo implementation, not a production vector DB study

These are acceptable because the main goal here is to show where the latency pressure is and why phase-specific memory traces matter.

## 21. Recommended Citation Targets Inside This Package

If you want to explain this version quickly in the paper or notes, the best entry points are:

- [README.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/README.md)
- [phase_aware_real_agent_v3_experiment_report_20260616.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/phase_aware_real_agent_v3_experiment_report_20260616.md)
- [phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml)
- [phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml)

