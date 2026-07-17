# Task Quality Curve Report

## Scope

This report summarizes the formal V5.1 fixed-subset Qwen task-quality curve on real MemoryCD data.

- Protocols: `official_single`, `official_cross`, `causal_cross`
- Tasks: `rating_prediction`, `item_ranking`
- Methods per group: `11`
- Effective evaluation groups: `264`
- Effective events per group: `180` or `200`

Artifacts:

- Summary CSV: `versions/20260622_memorycd_candidate_frontier_v5_1_formal/reports/task_quality_curve_summary.csv`
- Pareto JSON: `versions/20260622_memorycd_candidate_frontier_v5_1_formal/reports/pareto_budget_report.json`

## Main Findings

### Rating prediction

`rating_prediction` is largely insensitive to larger rerank budgets.

- `official_single`: best aggregate RMSE is `1.6197` with `recency_k8`
- `official_cross`: methods are effectively tied at `1.6476`
- `causal_cross`: methods are effectively tied at `1.6325`

Selection cost still rises materially with budget:

- `recency_k8`: near `0 ms` p50 selection
- `dense_k8`: about `12-13 ms` p50 selection
- `dense_r128_b64_k8`: about `76-80 ms` p50 selection

Interpretation:

- extra rerank budget is hard to justify for rating tasks
- this is direct evidence for budget compression on the selection/rerank path

### Item ranking

`item_ranking` benefits from better candidate frontier quality.

Aggregate best results by protocol:

- `official_single`: best `0.9732`, vs `0.9598` for `recency_k8`
- `official_cross`: best `0.9854`, vs `0.9689` for `recency_k8`
- `causal_cross`: best `0.9866`, vs `0.9778` for `recency_k8`

Representative setting-level gains:

- `official_cross / cross_to_beauty_and_personal_care`: `+0.0517` over recency
- `official_single / single_beauty_and_personal_care`: `+0.0283` over recency

Interpretation:

- candidate frontier quality matters for ranking
- larger budgets do not always win
- some low-budget methods already reach near-reference quality

## Pareto-Min-Budget Outcome

Across `24` protocol-task-setting groups:

- `rating_prediction`: `0 / 12` Pareto-min-budget selections
- `item_ranking`: `4 / 12` Pareto-min-budget selections

Observed Pareto groups:

- `causal_cross / item_ranking / cross_to_books`
- `official_cross / item_ranking / cross_to_beauty_and_personal_care`
- `official_cross / item_ranking / cross_to_books`
- `official_single / item_ranking / single_books`

This shows budget compression exists, but it is task-aware and setting-aware rather than universal.

## Hygiene Note

The early smoke trace `official_single_task_curve_subset.jsonl` has been moved to:

- `runs/task_curve_subset/_legacy_smoke/official_single_task_curve_subset.jsonl`

The summary script now excludes such legacy traces from the formal report.
