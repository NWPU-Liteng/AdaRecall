# Method-1 Motivation: MedMemoryBench v3

This package is a self-contained motivation experiment package for Method-1 on the English text-only MedMemoryBench workload.

It is intentionally scoped to motivation evidence only:

- M0: pipeline and dataset audit
- M1: anchor-preserving nested-history profiling
- M2: flat frontier budget study
- M3: chronological tile scattering
- O1: controller-defined execution-stage locality

It does not evaluate the proposed hypergraph layout, PIM runtime, semantic placement policy, replication policy, or online remapping.

## Current Status

See `STATUS.md`. Formal runs are gated on official MedMemoryBench acquisition, schema audit, GPU availability, and CUDA-enabled local model execution.

## Required Environment

Run from this directory:

```bash
conda activate agent-memory-profiling
export PYTHONHASHSEED=20260704
export CUDA_VISIBLE_DEVICES=0
```

The scripts fail loudly if formal GPU runs are attempted without CUDA.

## Pipeline Order

```bash
python scripts/probe_medmemorybench_schema.py \
  --data-root ../../../data/external/MedMemoryBench \
  --language en \
  --output-root .

python scripts/build_medmemorybench_memory.py \
  --config configs/data_medmemorybench_en.yaml \
  --granularity turn_pair \
  --output aggregate/memory_items_turn_pair.jsonl

python scripts/build_evidence_mapping.py \
  --memory-items aggregate/memory_items_turn_pair.jsonl \
  --queries aggregate/queries_medmemorybench_en.jsonl \
  --output-root .

python scripts/run_m1_history_scale.py \
  --config configs/m1_history_scale.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke

python scripts/run_m2_frontier_budget.py \
  --config configs/m2_budget.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke

python scripts/run_m3_tile_scattering.py \
  --config configs/m3_tile_scattering.yaml \
  --m2-log aggregate/M2_frontier_budget_invocations.jsonl

python scripts/run_o1_stage_trace.py \
  --config configs/o1_stage_locality.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke

python scripts/compute_o1_null_models.py \
  --stage-trace aggregate/O1_formal_stage_trace.jsonl \
  --num-permutations 1000 \
  --seed 20260704

python scripts/make_motivation_figures.py --root .
python scripts/summarize_medmemorybench_motivation.py --root .
python scripts/validate_medmemorybench_motivation.py --root .
```

## Claim Boundary

All reports must use `controller-defined execution stage` or `instrumented stage label` for O1. Do not call these labels benchmark-provided phases or ground-truth cognitive phases.
