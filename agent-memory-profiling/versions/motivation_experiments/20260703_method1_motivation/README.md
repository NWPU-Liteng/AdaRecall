# Method-1 Motivation Package

This directory consolidates the historical real-GPU motivation assets into a single guide-shaped package.

## Inputs
- V3 cross-encoder formal runs: `versions/20260616_phase_aware_real_agent_v3_formal/`
- V4 real-trace adaptive formal runs: `versions/20260621_real_trace_adaptive_phase_v4_formal/`

## Generated Outputs
- `raw/`: merged M1 and O1 JSONL files plus M3 analytical rows
- `aggregate/`: M1/M2/M3/O1 CSV summaries
- `figures/`: PNG and PDF figures
- `manifests/`: dataset and source-run manifests
- `runs/`: fresh V4 official-QA budget-grid smoke/formal reruns for `locomo` and `longmemeval_s`
- `RESULTS.md` and `LIMITATIONS.md`

## Current Status
- The formal V4 official-QA grid has been fully rerun at budgets `{8,16,32,64}` for both `locomo` and `longmemeval_s`.
- The QA path now uses metadata-aware prompting plus deterministic final-answer contraction, so the official QA metrics are no longer degenerate at zero.
- The package now closes the motivation QA loop at an auditable end-to-end level, but the formal task-score-vs-budget trend is still weak and non-monotonic; see `RESULTS.md` and `LIMITATIONS.md` before using the numbers as method-support evidence.

## Regeneration
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agent-memory-profiling
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/run_motivation_suite.py
```

After regeneration, validate with:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/validate_motivation_outputs.py \
  --root agent-memory-profiling/versions/20260703_method1_motivation
```

## Fresh V4 Official-QA Grid

Build the budget-grid configs:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/build_v4_official_qa_budget_grid.py
```

Run a smoke or formal sweep on real GPUs:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/run_v4_official_qa_budget_grid.py \
  --tier formal \
  --dataset locomo \
  --gpu 0
```

Summarize the completed runs:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/summarize_v4_official_qa_budget_grid.py
```

If the QA scoring logic changes, backfill existing run directories before summarizing:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/backfill_v4_official_qa_metrics.py
```

Analyze auditable query strata on the formal budget grid:

```bash
python agent-memory-profiling/versions/20260703_method1_motivation/scripts/analyze_v4_official_qa_strata.py
```
