# Status

Updated: 2026-07-05

## Completed in this package

- v3 package layout created under `agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3`.
- Config, schema, audit, memory construction, evidence mapping, experiment, figure, summary, and validation entry points are provided.
- The implementation preserves the v3 claim boundary: no hypergraph/PIM method evaluation is included.
- Official English MedMemoryBench files were loaded from `data/external/MedMemoryBench`.
- CPU-side M0 schema, memory-item, split, and evidence mapping artifacts were generated.
- M1 formal completed: 1688 invocation rows, 240/240 formal queries.
- M2 formal completed: 960 invocation rows, 240/240 formal queries across frontier budgets 8/16/32/64.
- M3 formal completed from fresh M2 budget-64 selections: 2160 scattering rows, 240/240 formal queries.
- O1 controller was tightened after pilot diagnostics; formal completed with 480 stage rows, 120/120 formal queries, and 479/480 valid parsed intents.
- O1 formal stage-shuffle null completed with 1000 permutations.
- Figures, final summary, and validation artifacts were regenerated after formal O1.
- Final validation passed in `reports/VALIDATION.md`.

## Current caveats

- The source Git repository revision is still pending because no `external/MedMemoryBench` Git clone was found; dataset files and local provenance are present.
- Evidence mapping uses a conservative session-level anchor. Evidence Recall@5 is therefore low and Evidence Completion@5 is zero for M2; do not interpret this as task failure alone.
- M2 answer generations are noisy and should not be used to claim answer-quality gains with larger frontier budgets.
- O1 formal locality is statistically strong overall, but the largest same-minus-cross contribution comes from the Plan stage.
- For O1 runs on 2026-07-05, `LOCAL_QWEN_CUDA_DEVICE=6` was used because setting `CUDA_VISIBLE_DEVICES=<idx>` made CUDA unavailable in the managed command environment. The O1 manifests record this deviation.

## Formal command sequence completed

The completed formal sequence was:

```bash
cd agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3
conda run -n agent-memory-profiling python scripts/run_m3_tile_scattering.py --config configs/m3_tile_scattering.yaml --m2-log aggregate/M2_frontier_budget_invocations.jsonl
LOCAL_QWEN_CUDA_DEVICE=6 conda run -n agent-memory-profiling python scripts/run_o1_stage_trace.py --config configs/o1_stage_locality.yaml --split configs/query_split_medmemorybench_en_v1.json --mode formal
conda run -n agent-memory-profiling python scripts/compute_o1_null_models.py --stage-trace aggregate/O1_formal_stage_trace.jsonl --num-permutations 1000 --seed 20260704
conda run -n agent-memory-profiling python scripts/make_motivation_figures.py --root .
conda run -n agent-memory-profiling python scripts/summarize_medmemorybench_motivation.py --root .
conda run -n agent-memory-profiling python scripts/validate_medmemorybench_motivation.py --root .
```
