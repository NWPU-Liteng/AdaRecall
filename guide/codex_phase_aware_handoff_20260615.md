# Codex Handoff: Phase-Aware Agent Memory Motivation Experiment

## 1. User Goal

The user is working on a paper about:

- building hypergraphs / multi-view memory layouts from **real agent execution traces**
- exploiting **phase-specific locality** in agent memory access
- designing **PIM-oriented primitives and layout optimizations**
- accelerating the parts of the agent memory pipeline that matter most, especially:
  - retrieval-related stages
  - candidate expansion / aggregation
  - rerank

Important clarification from the user:

- The agent's execution phases must be preserved.
- `planning`, `reasoning`, and `output` are not optional fluff; they are central to the paper's design motivation.
- The user does **not** want a toy search-only pipeline to replace the real agent path.

## 2. Key Methodology Decision Reached

We explicitly moved away from using a synthetic "multiply single selection by LLM rounds" default metric.

The current intended evidence chain is:

1. `shared_once` baseline
   - conservative real-agent baseline
   - one memory selection pass shared across later phases

2. `phase_aware` mode
   - planning / reasoning / output each get their own memory selection pass
   - records real per-phase memory traces
   - supports phase-locality analysis

3. `phase_trace` replay profiling
   - take real phase queries from agent traces
   - replay alternative selection baselines:
     - `retrieval_only`
     - `retrieval_plus_expansion`
     - `full_selection`
     - `oracle_pruned_rerank`
   - use this to prove:
     - retrieval itself may be small
     - rerank dominates selection cost
     - candidate count before rerank drives latency
     - pruning before rerank gives measurable speedup potential

This is now the preferred methodology.

## 3. What Was Changed

### 3.1 Agent execution can now consume phase-specific memory contexts

Modified:

- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)

Changes:

- `execute_agent_response(...)` now accepts:
  - `phase_contexts`
  - `phase_selected`
- `_run_llm_agent(...)` now uses:
  - planning-specific context
  - reasoning-specific context
  - output-specific context
- Tool-inspection stage can now use phase-specific selected memories.

This makes `phase_aware` mode semantically meaningful instead of being only bookkeeping.

### 3.2 Motivation profiling now supports `agent_memory_mode`

Modified:

- [profile_longgraph_memory_selection.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

Important additions:

- `agent_memory_mode` with supported values:
  - `shared_once`
  - `phase_aware`

Behavior:

- `shared_once`
  - one selection pass
  - later agent phases share the same context

- `phase_aware`
  - planning / reasoning / output each issue their own selection pass
  - phase trace rows are emitted
  - per-phase selection latencies are aggregated into:
    - `phase_selection_total_latency_ms`
    - `phase_retrieval_total_latency_ms`
    - `phase_rerank_total_latency_ms`
  - per-phase prompt / selected memories are passed to `agent_executor`

Important helper added:

- `run_selection_profile_pass(...)`
  - reusable selection-only profiling pass
  - used later by replay scripts

### 3.3 Output layout now includes `phase_trace.jsonl`

Modified:

- [trace_logger.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/trace_logger.py)

New output path:

- `phase_trace.jsonl`

### 3.4 Summary logic was rewritten toward phase-aware evidence

Modified:

- [metrics.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/metrics.py)
- [summarize_agent_latency_suite.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/summarize_agent_latency_suite.py)

Current summary fields emphasize:

- `selection_total_latency_ms`
- `phase_selection_total_latency_ms`
- `phase_retrieval_total_latency_ms`
- `phase_rerank_total_latency_ms`
- `phase_selection_to_total_response_ratio`
- `phase_retrieval_to_total_response_ratio`
- `phase_rerank_to_total_response_ratio`
- `phase_rerank_ratio`

The suite logic also infers `selection_invocation_count` from:

- `shared_once -> 1`
- `phase_aware -> 3`

### 3.5 The old synthetic default was removed from `real_agent_v2`

Modified:

- [build_agent_latency_suite.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/build_agent_latency_suite.py)
- all `real_agent_v2` suite configs under:
  - [configs/agent_latency_suite_20260615](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/configs/agent_latency_suite_20260615)

Important state now:

- `real_agent_v2` default config uses:
  - `agent_memory_mode = "shared_once"`
- old
  - `selection_invocation_strategy = "per_llm_round"`
  has been removed from the active defaults

### 3.6 Replay script was added

Added:

- [replay_phase_traces.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/replay_phase_traces.py)

Purpose:

- read `phase_trace.jsonl`
- reconstruct `QuerySpec`
- replay:
  - `retrieval_only`
  - `retrieval_plus_expansion`
  - `full_selection`
  - `oracle_pruned_rerank`
- write replay outputs:
  - `raw_profile.jsonl`
  - summary
  - plots

## 4. README Updates

Modified:

- [README.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/README.md)

Now documents:

- `agent_memory_mode`
- `phase_trace.jsonl`
- replay command example

## 5. Validation Work Already Done

### 5.1 Static validation

The following passed `py_compile`:

- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)
- [trace_logger.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/trace_logger.py)
- [profile_longgraph_memory_selection.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)
- [metrics.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/metrics.py)
- [plot_results.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/plot_results.py)
- [summarize_agent_latency_suite.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/summarize_agent_latency_suite.py)
- [build_agent_latency_suite.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/build_agent_latency_suite.py)
- [replay_phase_traces.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/replay_phase_traces.py)

### 5.2 Small methodology validation run

Added config:

- [phase_aware_method_validation_locomo_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/configs/phase_aware_method_validation_locomo_20260615.yaml)

This validation config intentionally used:

- `agent_memory_mode = "phase_aware"`
- `reranker_backend = "cross_encoder"`
- `reranker_batch_size = 1`
- `embedding_backend = "hash"`
- `agent_execution_backend = "calibrated_mock"`

Reason:

- Current session could not access CUDA, so real local Qwen generation / embedding could not run.
- This run was used to validate the **method chain**, not the final GPU-backed paper result.

Produced outputs:

- [raw_profile.jsonl](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/outputs/motivation_memory_selection_phase_aware_method_validation_locomo_20260615/raw_profile.jsonl)
- [phase_trace.jsonl](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/outputs/motivation_memory_selection_phase_aware_method_validation_locomo_20260615/phase_trace.jsonl)

Observed from the raw data:

- `rows = 8` in `raw_profile.jsonl`
- `rows = 24` in `phase_trace.jsonl`
- `selection_invocation_count = 3` under `phase_aware`

Example signals already observed:

- `relation_0001 full_selection`
  - `phase_selection_total_latency_ms ≈ 3115 ms`
  - `phase_rerank_total_latency_ms ≈ 3095 ms`
  - `agent_execution_total_latency_ms ≈ 27.6 ms`

- `plan_continuity_0001 full_selection`
  - `phase_selection_total_latency_ms ≈ 4687 ms`
  - `phase_rerank_total_latency_ms ≈ 4672 ms`
  - `agent_execution_total_latency_ms ≈ 27.3 ms`

Interpretation:

- under `phase_aware + cross_encoder + batch_size=1`
- selection becomes clearly nontrivial
- rerank dominates selection almost completely

### 5.3 Replay validation also worked

Replay output:

- [raw_profile.jsonl](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/outputs/motivation_memory_selection_phase_aware_method_validation_locomo_20260615_replay/raw_profile.jsonl)

Observed replay aggregation:

- `planning full_selection`
  - `avg_selection_ms ≈ 1225`
  - `avg_rerank_ms ≈ 1223`
  - `avg_num_before_rerank ≈ 81.6`

- `planning oracle_pruned_rerank`
  - `avg_selection_ms ≈ 749`
  - `avg_rerank_ms ≈ 745`
  - `avg_num_before_rerank = 50`

- `reasoning full_selection`
  - `avg_selection_ms ≈ 1281`
  - `avg_rerank_ms ≈ 1279`
  - `avg_num_before_rerank ≈ 83.5`

- `reasoning oracle_pruned_rerank`
  - `avg_selection_ms ≈ 741`
  - `avg_rerank_ms ≈ 736`
  - `avg_num_before_rerank = 50`

- `output full_selection`
  - `avg_selection_ms ≈ 1192`
  - `avg_rerank_ms ≈ 1190`
  - `avg_num_before_rerank ≈ 80.8`

- `output oracle_pruned_rerank`
  - `avg_selection_ms ≈ 748`
  - `avg_rerank_ms ≈ 744`
  - `avg_num_before_rerank = 50`

Also:

- `retrieval_only` and `retrieval_plus_expansion` were only around `2.7 ms` to `3.7 ms`

This is already strong support for:

- coarse retrieval itself can be small
- rerank dominates selection cost
- reducing candidates before rerank materially reduces selection latency

## 6. Known Issues / Blockers

### 6.1 Current session cannot see CUDA

Direct checks performed in this conversation:

- `nvidia-smi` failed with:
  - `couldn't communicate with the NVIDIA driver`
- `torch.cuda.is_available() == False`
- `torch.cuda.device_count() == 0`

Important nuance:

- This does **not** prove the server has no free GPU.
- It only proves that **this Codex execution environment / session** cannot currently access CUDA.

This blocked:

- `LocalQwenLLMClient`
- `LocalQwenEmbeddingClient`
- `TransformerEmbeddingClient`

All currently throw runtime errors because their implementations require GPU.

### 6.2 `tabulate` missing in environment

Both the main profiling run and replay can fail at summary generation because:

- `pandas.DataFrame.to_markdown()` requires optional dependency `tabulate`

Raw outputs were still written successfully before that point.

This should be fixed in code, not worked around manually every time.

## 7. Recommended Next Steps

### Priority A: Make summaries robust without `tabulate`

Modify:

- [metrics.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/metrics.py)

Goal:

- wrap `summary.to_markdown(...)` in a fallback
- if `tabulate` missing:
  - write plain text summary
  - or write a CSV pointer + simple bullet conclusions

This is the cleanest next step.

### Priority B: Add replay-specific summary and plots

Current replay raw file already contains useful evidence, but needs a friendlier layer.

Add:

- replay summary aggregation by:
  - `phase_name`
  - `baseline_mode`
- report:
  - `avg_num_before_rerank`
  - `avg_selection_ms`
  - `avg_rerank_ms`
  - `avg_rerank_ratio`
  - `speedup vs full_selection`

Potential plot ideas:

- per-phase rerank latency bar chart
- per-phase candidates-before-rerank chart
- oracle-pruned speedup by phase

### Priority C: Run the same method with real GPU-backed agent execution

Once CUDA is actually visible to the executing session:

- use a GPU-capable config
- likely based on:
  - [phase_aware_real_agent_v2_locomo_validation_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/configs/phase_aware_real_agent_v2_locomo_validation_20260615.yaml)
- switch to:
  - real `local_qwen` generation
  - either `local_qwen` or `transformer` embedding

This would convert the methodology validation into a proper real-agent validation.

## 8. Commands Used / Useful Commands

### GPU visibility checks already run

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

### Methodology validation run

```bash
/data2/user25263093/anaconda3/envs/agent-memory-profiling/bin/python \
  agent-memory-profiling/experiments/motivation_memory_selection/run_motivation_profile.py \
  --config agent-memory-profiling/experiments/motivation_memory_selection/configs/phase_aware_method_validation_locomo_20260615.yaml
```

### Replay run

```bash
/data2/user25263093/anaconda3/envs/agent-memory-profiling/bin/python \
  agent-memory-profiling/experiments/motivation_memory_selection/replay_phase_traces.py \
  --config agent-memory-profiling/experiments/motivation_memory_selection/configs/phase_aware_method_validation_locomo_20260615.yaml \
  --phase-trace agent-memory-profiling/outputs/motivation_memory_selection_phase_aware_method_validation_locomo_20260615/phase_trace.jsonl \
  --output-dir agent-memory-profiling/outputs/motivation_memory_selection_phase_aware_method_validation_locomo_20260615_replay
```

## 9. Suggested Immediate Follow-Up for Next Codex Session

If starting a new Codex CLI conversation, the best first tasks are:

1. Fix `tabulate` dependency fallback in summary generation.
2. Add replay-specific summary / plots.
3. If possible, restore GPU visibility in the executing session and rerun the real-agent phase-aware validation.

If GPU becomes available, the next run should focus on:

- `phase_aware`
- `cross_encoder`
- `batch_size = 1`
- `relation`, `plan_continuity`, `constraint`
- then replay `full_selection` vs `oracle_pruned_rerank`

