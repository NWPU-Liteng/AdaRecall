# Motivation Memory Selection Experiment

This experiment profiles agent memory selection stages rather than only
end-to-end latency. It is designed to answer whether long-term memory pressure
mainly appears in candidate expansion, aggregation, rerank load, and host-side
traffic instead of coarse retrieval alone.

## Entry point

Run the default configuration:

```bash
python experiments/motivation_memory_selection/run_motivation_profile.py \
  --config experiments/motivation_memory_selection/configs/motivation_default.yaml
```

Run the quick validation path:

```bash
python experiments/motivation_memory_selection/run_motivation_profile.py \
  --config experiments/motivation_memory_selection/configs/motivation_default.yaml \
  --quick
```

## Public Datasets

Convert a public dataset into the unified memory JSONL schema first, then point
`input_memory_paths` at the converted file.

Example for `LoCoMo`:

```bash
python experiments/motivation_memory_selection/import_public_dataset.py \
  --dataset locomo \
  --input /path/to/locomo10.json \
  --output data/motivation_memory_selection/locomo_memories_1000.jsonl
```

Example for `MSC` after exporting a JSON/JSONL file from ParlAI:

```bash
python experiments/motivation_memory_selection/import_public_dataset.py \
  --dataset msc \
  --input /path/to/msc_export.json \
  --output data/motivation_memory_selection/msc_memories_5000.jsonl
```

Example for `LongMemEval`:

```bash
python experiments/motivation_memory_selection/import_public_dataset.py \
  --dataset longmemeval \
  --input /path/to/longmemeval_history.json \
  --output data/motivation_memory_selection/longmemeval_memories_1000.jsonl
```

Then run the profiler with the matching template config:

```bash
python experiments/motivation_memory_selection/run_motivation_profile.py \
  --config experiments/motivation_memory_selection/configs/motivation_locomo_template.yaml
```

## Outputs

The run writes results to `outputs/motivation_memory_selection/`:

- `raw_profile.jsonl`
- `phase_trace.jsonl`
- `summary.csv`
- `summary.md`
- `figures/*.png`
- `logs/run_config.json`
- `logs/run_result.json`

To replay real agent phase queries with alternative selection settings:

```bash
python experiments/motivation_memory_selection/replay_phase_traces.py \
  --config experiments/motivation_memory_selection/configs/motivation_locomo_cross_encoder_agent_medium_20260615.yaml \
  --phase-trace outputs/motivation_memory_selection_locomo_cross_encoder_agent_medium_20260615/phase_trace.jsonl \
  --output-dir outputs/motivation_memory_selection_locomo_cross_encoder_agent_medium_20260615_replay
```

This replay mode keeps the phase queries from a real agent run, but re-executes
the selection pipeline under different baselines such as `retrieval_only`,
`full_selection`, and `oracle_pruned_rerank`.

## Notes

- The config file is stored as JSON-compatible YAML so it can be parsed without
  an extra YAML dependency.
- The default path uses hash embeddings and mock generation, which keeps the
  experiment offline-safe.
- `reranker_backend` supports:
  - `heuristic`: the lightweight in-repo scoring rule used for the original profiling runs.
  - `cross_encoder`: the mainstream two-stage setup where retrieval returns candidates and a sequence-classification reranker rescoring step reorders them.
- To use real memory data, add `input_memory_paths` in the config as a mapping
  from declared `memory_size` to a JSONL path. Each row should preferably
  contain `memory_id`, `text` or `content`, `timestamp` or `time`,
  `memory_type`, `task_tags` or `module_tags`, `entity_tags`, and optional
  `relations` and `gold_group_id`.
- `cross_encoder` requires a local HuggingFace-compatible sequence-classification
  reranker model directory. A causal LM directory such as a standard Qwen
  generation checkpoint is not a valid replacement. The provided
  `motivation_cross_encoder_template_20260615.yaml` shows the required fields:
  `reranker_model_dir`, `reranker_batch_size`, and `reranker_max_length`.
- `agent_memory_mode` controls how memory is invoked during real agent execution:
  - `shared_once`: the conservative baseline. One selection pass is built and shared by later agent phases.
  - `phase_aware`: planning, reasoning, and output phases each issue their own memory selection pass and emit rows into `phase_trace.jsonl`.
- `phase_trace.jsonl` is intended for replay-oriented analysis. It lets you reuse
  real agent phase queries to compare `retrieval_only`, `full_selection`, and
  `oracle_pruned_rerank` without changing the downstream agent structure.
- `MSC` support here assumes you export the dataset into JSON/JSONL first. The
  official source is the ParlAI `MSC` task page: https://parl.ai/projects/msc/
- `LoCoMo` official source: https://github.com/snap-research/locomo
- `LongMemEval` official source: https://github.com/xiaowu0162/LongMemEval
