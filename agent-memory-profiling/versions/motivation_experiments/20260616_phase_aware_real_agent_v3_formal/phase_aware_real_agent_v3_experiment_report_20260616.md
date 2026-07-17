# Phase-Aware Real-Agent V3 Experiment Report

## Scope

- Agent memory mode: `phase_aware`
- Agent execution backend: local Qwen (`generation_backend=local_qwen`, `agent_execution_backend=llm`)
- Embedding backend: BGE transformer (`models/bge-base-en-v1.5`)
- Compared rerank backends: `heuristic` and `cross_encoder`
- Datasets: `LoCoMo`, `LongMemEval`, `MSC`
- Compared baselines inside each run: `full_selection` and `oracle_pruned_rerank`

## Formal Run Map

- `locomo/heuristic`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615`
- `locomo/cross_encoder`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615`
- `longmemeval/heuristic`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616`
- `longmemeval/cross_encoder`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616`
- `msc/heuristic`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616`
- `msc/cross_encoder`: active config `experiments/motivation_memory_selection/configs/phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616.yaml`, config snapshot `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616.yaml`, outputs `versions/20260616_phase_aware_real_agent_v3_formal/runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616`

## Overall Summary

| Dataset | Rerank | Raw Rows | Phase Rows | Query Types | Phase Selection ms | Retrieval-like ms | Rerank ms | Agent ms | Total Response ms | Selection Share | Retrieval Share | Rerank Share | Selection Calls | Avg Before Rerank |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|
| locomo | heuristic | 400 | 1200 | constraint, fact, plan_continuity, relation, temporal | 75.2 | 73.6 | 1.6 | 1854.9 | 1930.1 | 3.9% | 3.8% | 0.1% | 3.0 | 69.05 |
| locomo | cross_encoder | 400 | 1200 | constraint, fact, plan_continuity, relation, temporal | 1611.7 | 44.0 | 1567.7 | 1758.7 | 3370.4 | 44.3% | 1.5% | 42.8% | 3.0 | 69.05 |
| longmemeval | heuristic | 800 | 2400 | constraint, fact, plan_continuity, relation, temporal | 92.7 | 90.9 | 1.8 | 5981.8 | 6074.4 | 1.5% | 1.5% | 0.0% | 3.0 | 73.64 |
| longmemeval | cross_encoder | 800 | 2400 | constraint, fact, plan_continuity, relation, temporal | 3829.9 | 50.0 | 3780.0 | 5761.4 | 9591.3 | 36.8% | 0.5% | 36.3% | 3.0 | 73.64 |
| msc | heuristic | 800 | 2400 | constraint, fact, plan_continuity, relation, temporal | 332.1 | 330.4 | 1.8 | 1721.1 | 2053.3 | 15.8% | 15.7% | 0.1% | 3.0 | 72.73 |
| msc | cross_encoder | 800 | 2400 | constraint, fact, plan_continuity, relation, temporal | 1820.5 | 228.9 | 1591.6 | 1622.8 | 3443.3 | 49.8% | 7.1% | 42.7% | 3.0 | 72.73 |

## Baseline Breakdown

| Dataset | Rerank | Baseline | Total ms | Phase Selection ms | Phase Rerank ms | Agent ms | Selection Share | Rerank Share | Avg Before Rerank |
|:--|:--|:--|--:|--:|--:|--:|:--|:--|--:|
| locomo | heuristic | full_selection | 1926.3 | 79.9 | 2.2 | 1846.4 | 4.2% | 0.1% | 92.73 |
| locomo | heuristic | oracle_pruned_rerank | 1933.9 | 70.4 | 0.9 | 1863.5 | 3.6% | 0.0% | 45.38 |
| locomo | cross_encoder | full_selection | 3856.1 | 2089.5 | 2044.7 | 1766.6 | 50.2% | 48.9% | 92.73 |
| locomo | cross_encoder | oracle_pruned_rerank | 2884.7 | 1133.9 | 1090.6 | 1750.8 | 38.3% | 36.7% | 45.38 |
| longmemeval | heuristic | full_selection | 6149.0 | 98.4 | 2.5 | 6050.6 | 1.6% | 0.0% | 99.58 |
| longmemeval | heuristic | oracle_pruned_rerank | 5999.9 | 87.0 | 1.1 | 5912.9 | 1.4% | 0.0% | 47.69 |
| longmemeval | cross_encoder | full_selection | 10949.0 | 5157.6 | 5103.7 | 5791.4 | 43.7% | 43.2% | 99.58 |
| longmemeval | cross_encoder | oracle_pruned_rerank | 8233.5 | 2502.2 | 2456.2 | 5731.3 | 30.0% | 29.4% | 47.69 |
| msc | heuristic | full_selection | 2074.8 | 344.2 | 2.5 | 1730.6 | 16.1% | 0.1% | 98.75 |
| msc | heuristic | oracle_pruned_rerank | 2031.7 | 320.1 | 1.0 | 1711.6 | 15.5% | 0.1% | 46.72 |
| msc | cross_encoder | full_selection | 3915.0 | 2295.4 | 2065.4 | 1619.6 | 54.9% | 48.5% | 98.75 |
| msc | cross_encoder | oracle_pruned_rerank | 2971.6 | 1345.6 | 1117.7 | 1626.0 | 44.6% | 36.9% | 46.72 |

## Main Findings

- Under `heuristic` rerank, phase-aware memory selection remains light on `LoCoMo` and `LongMemEval`, and becomes moderately visible on `MSC` because retrieval/filter/aggregation scales with the larger memory pool.
- Under `cross_encoder` rerank, phase-aware memory selection becomes a first-order cost on all three datasets, with rerank dominating the selection path.
- `oracle_pruned_rerank` consistently reduces `num_before_rerank`, and this reduction materially lowers total response time only when the rerank backend is heavy enough for rerank to matter.
- The cleanest paper-facing evidence for `selection/rerank` as the primary optimization target comes from the `cross_encoder` runs, not the `heuristic` runs.

## Versioning

- This report is stored under `versions/20260616_phase_aware_real_agent_v3_formal`.
- Formal run directories are versioned under `versions/20260616_phase_aware_real_agent_v3_formal/runs`.
- Formal config snapshots are stored under `versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot`.
- Older, non-formal, mock-based, quick, medium, replay, and pre-v3 validation outputs were moved to the archive subtree to reduce root-level clutter.
