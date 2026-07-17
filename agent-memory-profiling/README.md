# Agent Memory Profiling

Minimal, offline-safe profiling experiment for agent memory invocation latency.

## Canonical experiment package

All experiment versions are consolidated under
[the v21 package](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260714_method1_fsa_dynamic_hypergraph_v21).
Read its [consolidation note](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260714_method1_fsa_dynamic_hypergraph_v21/CONSOLIDATION_20260717.md)
for the canonical inputs and historical provenance map. The former June
archives are compacted under `versions/20260714_method1_fsa_dynamic_hypergraph_v21/provenance/legacy_versions/`.

## Quick start

```bash
chmod +x scripts/preflight.sh scripts/setup_env.sh scripts/smoke_test.sh
./scripts/preflight.sh
./scripts/setup_env.sh
./scripts/smoke_test.sh
```

If smoke test succeeds, continue with larger memory sizes as described in
`../codex_agent_memory_profiling_guide.md`.

## Profiling modes

The profiler now supports both a single retrieval pass and an agent-style
multi-round retrieval mode. For experiments where a real local LLM makes the
memory stage look artificially small, compare:

```bash
python scripts/run_profile.py \
  --memory-sizes 100000 \
  --candidate-k 50 100 200 \
  --top-k 5 10 \
  --repeat 5 \
  --execution-mode single \
  --memory-rounds 1 \
  --llm-backend local_qwen \
  --embedding-backend local_qwen

python scripts/run_profile.py \
  --memory-sizes 100000 \
  --candidate-k 50 100 200 \
  --top-k 5 10 \
  --repeat 5 \
  --execution-mode agent \
  --memory-rounds 4 \
  --llm-max-new-tokens 8 \
  --llm-backend local_qwen \
  --embedding-backend local_qwen
```

The output CSV now records `execution_mode`, `memory_rounds`,
`num_memory_calls`, and `t_memory` so single-pass and agent-style runs can be
compared directly.

## Retrieval backends

The project now supports three retrieval paths:

- `numpy`
  - Exact dense-vector scan over local `.npy` files
- `ann_numpy`
  - Local two-stage ANN retrieval using coarse centroid routing plus exact
    scoring inside probed clusters
- `qdrant`
  - External Qdrant vector database when available

To build and run the new local ANN retrieval path without any external service:

```bash
python scripts/build_index.py \
  --input data/synthetic_memories_100000.jsonl \
  --prefer-qdrant 0 \
  --retrieval-backend ann_numpy \
  --embedding-backend local_qwen

python scripts/run_profile.py \
  --memory-sizes 100000 \
  --candidate-k 50 100 200 \
  --top-k 5 10 \
  --repeat 5 \
  --prefer-qdrant 0 \
  --retrieval-backend ann_numpy \
  --execution-mode langgraph \
  --memory-rounds 4 \
  --llm-max-new-tokens 8 \
  --llm-backend local_qwen \
  --embedding-backend local_qwen
```

## LangGraph experiment

If you want the agent path to be implemented with an existing open-source
framework instead of the built-in orchestration, use `execution-mode=langgraph`.
This runs the memory loop inside a LangGraph `StateGraph`, while keeping the
same embedding, retrieval, rerank, packaging, and final generation components.

```bash
export LOCAL_QWEN_CUDA_DEVICE=0

python scripts/run_profile.py \
  --memory-sizes 100000 \
  --candidate-k 50 100 200 \
  --top-k 5 10 \
  --repeat 5 \
  --execution-mode langgraph \
  --memory-rounds 4 \
  --llm-max-new-tokens 8 \
  --llm-backend local_qwen \
  --embedding-backend local_qwen
```

The resulting CSV will tag these rows with `framework_mode=langgraph`.

## Local Qwen runtime

For the bundled `Qwen3-4B` directory in this repo, the validated runtime is:

- `torch >= 2.1.0`
- `transformers >= 4.51.0, < 4.52.0`
- `tokenizers >= 0.21.0, < 0.22.0`

Using `transformers 4.36.x` is too old for Qwen3. Using `transformers 5.x` may
require a newer PyTorch than the one currently used in this environment.

For the standard ANN-backed LangGraph comparison used in this repo, run:

```bash
chmod +x scripts/run_langgraph_experiment.sh
GPU_ID=0 MEMORY_SIZE=100000 REPEAT=5 ./scripts/run_langgraph_experiment.sh
```

This produces:

- `results/raw_profile_langgraph_4rounds_24tok.csv`
- `results/raw_profile_langgraph_4rounds_16tok.csv`
- `results/raw_profile_langgraph_4rounds_8tok.csv`

The script uses the currently activated environment's `python`, validates the
local Qwen runtime versions before starting, automatically prepares the local
ANN index when needed, and runs with `--retrieval-backend ann_numpy`.
