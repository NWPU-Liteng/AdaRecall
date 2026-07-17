# GPT Analysis Brief: Method-1 Motivation Experiments on MedMemoryBench

Updated: 2026-07-05

This document summarizes the experimental setting, run arrangement, formal results, and claim boundaries for the MedMemoryBench v3 motivation package. It is intended to be pasted into GPT or another analysis assistant to support paper-positioning, figure selection, result interpretation, and claim wording.

## 1. Package and Objective

Package path:

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3
```

Objective:

Build a reproducible, publication-oriented motivation package for Method-1 using the English text-only MedMemoryBench workload and local Qwen3-4B-based memory agent components. The package is designed to establish why structural/stage-aware memory organization is worth studying. It is not a final method evaluation.

The intended motivation chain is:

```text
Longitudinal personal-health histories accumulate many medically relevant,
revised, and distracting memory items.
        ->
A flat retriever must expand its candidate frontier to preserve evidence coverage,
but expansion increases materialization and reranking cost.
        ->
The selected high-value memories are scattered across chronological memory tiles,
so tile-granular near-memory execution would over-fetch irrelevant content.
        ->
Under a fixed, auditable text-only controller, different controller-defined
execution stages access different but recurrent memory subsets.
        ->
This motivates stage-conditioned structural memory organization.
```

Claim boundary:

This package does not evaluate the final hypergraph layout, PIM runtime, replication policy, online remapping, or dense-vs-graph-vs-hypergraph method comparison. It should not be used to claim final method superiority.

## 2. Data and Workload

Dataset:

- Official English MedMemoryBench files are loaded from:

```text
/data2/user25263093/userliteng/26_code/preTest/data/external/MedMemoryBench
```

Schema/workload facts:

- English queries extracted: 1939.
- Personas: 20.
- Query types: 6.
- `dialogues_with_noise` is available.
- Source files include stage-like clinical report fields (`phase_1` to `phase_5`), but these are not used as O1 execution-stage labels.

Memory construction:

- Memory granularity: chronological patient-doctor turn-pair memory item, plus one singleton.
- Total memory items: 42,034.
- Distribution:
  - `medical`: 15,988.
  - `noise_family`: 13,026.
  - `noise_health`: 13,020.

Evidence mapping:

- Evidence mapping coverage: 1,935/1,939 queries, 99.79%.
- Evidence anchor is conservative and session-level:

```text
query source_key_points.session_id
    -> all medical turn-pair memories for that persona/session
```

This is deliberately not a span-level key-point mapping. It makes Evidence Recall@5 conservative/low and makes Evidence Completion@5 usually zero because the session-level gold set is often much larger than top-5.

Important caveat:

The source Git repository revision is still pending because no local `external/MedMemoryBench` Git clone was present. The downloaded dataset files and schema/provenance artifacts are present.

## 3. Models and Retrieval Components

Main model/configuration:

- Controller/answer model: local Qwen3-4B path `../../../dir`.
- Embedding model: `../../models/bge-base-en-v1.5`.
- Reranker: `../../models/cross-encoder-ms-marco-MiniLM-L6-v2`.
- Retrieval style: flat retrieval with ANN overfetch, cross-encoder reranking, final top-k selection.
- Final top-k: 5 for the core M2/M3/O1 analyses.

GPU/runtime notes:

- M1 and M2 formal runs were completed earlier using `CUDA_VISIBLE_DEVICES=3`.
- O1 smoke/pilot/formal runs on 2026-07-05 used `LOCAL_QWEN_CUDA_DEVICE=6`.
- Reason: in this managed environment, setting `CUDA_VISIBLE_DEVICES=<idx>` made `torch.cuda.is_available()` false, while leaving `CUDA_VISIBLE_DEVICES` unset allowed CUDA initialization. The local Qwen, embedding, and reranker clients read `LOCAL_QWEN_CUDA_DEVICE`.
- O1 manifests record:
  - `cuda_visible_devices: null`.
  - `local_qwen_cuda_device: "6"`.

## 4. Experiments

| ID | Name | Purpose | Formal coverage | Main artifacts |
|---|---|---|---:|---|
| M0 | Pipeline and dataset audit | Validate data schema, memory construction, evidence mapping, and claim boundary | Full dataset audit | `reports/01_medmemorybench_schema_audit.md`, `reports/02_memory_item_and_evidence_audit.md` |
| M1 | History-scale profiling | Measure how history scale and official noise affect selection cost/state footprint | 240/240 queries | `aggregate/M1_history_scale_invocations.jsonl`, `aggregate/M1_history_scale_summary.csv` |
| M2 | Frontier budget study | Measure flat frontier cost versus evidence/oracle-overlap returns | 240/240 queries x 4 budgets | `aggregate/M2_frontier_budget_invocations.jsonl`, `aggregate/M2_frontier_budget_summary.csv`, `aggregate/M2_frontier_occupancy.csv` |
| M3 | Chronological tile scattering | Quantify over-fetch/scattering under chronological memory placement | 240/240 queries | `aggregate/M3_tile_scattering.csv` |
| O1 | Controller-defined stage locality | Test whether fixed controller stages induce reproducible non-random memory-access locality | 120/120 queries x 4 stages | `aggregate/O1_formal_stage_trace.jsonl`, `aggregate/O1_same_cross_stage_stats.csv`, `aggregate/O1_stage_shuffle_null.csv`, `aggregate/O1_pair_recurrence.csv` |

## 5. M1: History-Scale Profiling

Formal artifacts:

- `aggregate/M1_history_scale_invocations.jsonl`: 1688 rows.
- `aggregate/M1_history_scale_summary.csv`: 9 rows.
- Formal query coverage: 240/240.

Key observations:

- Medical-only full-history mean selection latency: 132.29 ms.
- Medical-plus-official-noise full-history mean selection latency: 134.25 ms.
- Frontier fill ratio stayed at 1.0 across reported scales.
- Host/index/payload byte footprint grows with larger history scales and with official noise.

Interpretation:

M1 supports the workload premise that longitudinal histories and distractors increase memory-state footprint and materialization/scanning burden for flat retrieval.

Caveat:

The latency curve is not strictly monotonic at every scale. Candidate composition and reranker behavior can dominate local latency differences. Do not claim a universal monotonic latency law from M1 alone.

## 6. M2: Flat Frontier Budget Study

Formal artifacts:

- `aggregate/M2_frontier_budget_invocations.jsonl`: 960 rows.
- `aggregate/M2_frontier_budget_summary.csv`: 24 rows.
- `aggregate/M2_frontier_occupancy.csv`: 960 rows.
- Formal query coverage: 240/240.

Budget means:

| Frontier budget | Selection ms | Rerank ms | Evidence Recall@5 | Evidence Completion@5 | RerankerOracleOverlap@5 | Answer generation ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 58.55 | 43.94 | 0.110 | 0.000 | 0.554 | 8315.16 |
| 16 | 97.91 | 84.51 | 0.110 | 0.000 | 0.720 | 8351.18 |
| 32 | 143.77 | 130.30 | 0.112 | 0.000 | 0.858 | 8329.48 |
| 64 | 229.23 | 215.75 | 0.111 | 0.000 | 1.000 | 8274.46 |

Interpretation:

M2 supports the cost/frontier/oracle-overlap motivation:

- Increasing the frontier budget increases selection and reranking cost.
- RerankerOracleOverlap@5 improves from 0.554 at budget 8 to 1.000 at budget 64.
- Evidence Recall@5 is low and mostly flat under the conservative session-level evidence anchor.
- Evidence Completion@5 is zero because the gold evidence set is usually much larger than final top-5.

Caveat:

Do not claim answer quality improves with larger frontier budgets. Raw answer generations are noisy and sometimes off target. M2 is best used as a cost/frontier/oracle-overlap result, not as a task-quality improvement result.

## 7. M3: Chronological Tile Scattering

Formal artifacts:

- `aggregate/M3_tile_scattering.csv`: 2160 rows.
- Formal query coverage: 240/240.
- Source selected memories: fresh M2 formal budget-64 top-5 selections.

Mean results:

| Tile capacity | Layout | Activated tiles | Tile utilization | Access amplification | Scatter ratio |
|---:|---|---:|---:|---:|---:|
| 8 | chronological | 4.50 | 0.142 | 7.21 | 4.50 |
| 8 | random_per_persona | 4.94 | 0.127 | 7.90 | 4.94 |
| 8 | oracle tile-packing lower bound | 1.00 | 0.625 | 1.60 | 1.00 |
| 16 | chronological | 4.21 | 0.078 | 13.48 | 4.21 |
| 16 | random_per_persona | 4.90 | 0.064 | 15.68 | 4.90 |
| 16 | oracle tile-packing lower bound | 1.00 | 0.313 | 3.20 | 1.00 |
| 32 | chronological | 3.84 | 0.045 | 24.56 | 3.84 |
| 32 | random_per_persona | 4.85 | 0.032 | 31.04 | 4.85 |
| 32 | oracle tile-packing lower bound | 1.00 | 0.156 | 6.40 | 1.00 |

Interpretation:

Chronological physical placement scatters jointly selected top-5 memories across about 3.84 to 4.50 tiles on average, while an oracle tile-packing lower bound would use one tile at these tested capacities. This supports a physical over-fetch/headroom motivation.

Caveat:

The oracle tile-packing lower bound is not a deployable method. M3 does not prove a specific layout or PIM design is superior.

## 8. O1: Controller-Defined Stage Locality

Formal artifacts:

- `aggregate/O1_formal_stage_trace.jsonl`: 480 rows.
- `aggregate/O1_same_cross_stage_stats.csv`: 4 rows.
- `aggregate/O1_stage_shuffle_null.csv`: 1000 permutations.
- `aggregate/O1_pair_recurrence.csv`: 4 rows.
- Formal query coverage: 120/120.
- Exactly four stages per query:
  - `ground`
  - `plan`
  - `reason`
  - `reflect`

Controller design:

- Stage label source: fixed controller FSM, `controller_fsm_v1`.
- Stages are assigned before retrieval.
- Stage prompts include query text/type/session and prior controller summaries only.
- Stage prompts do not include gold answers, source key points, or retrieved memory text before intent generation.
- Retrieval budgets are identical across stages.

Controller quality:

- Parsed intent JSON: 479/480 rows, 99.79%.
- Retrieval-goal prefix validity: 100%.
- Queries with more than one unique selected set across four stages: 120/120.
- Mean unique selected sets per query: 3.925/4.
- Mean intent generation latency: 7037.96 ms.
- Mean selection latency after intent generation: 147.57 ms.

Same-stage versus cross-stage overlap:

| Stage | Same-stage overlap | Cross-stage overlap | Same-minus-cross |
|---|---:|---:|---:|
| ground | 0.06099 | 0.04708 | 0.01391 |
| plan | 0.16155 | 0.04155 | 0.11999 |
| reason | 0.05127 | 0.04453 | 0.00675 |
| reflect | 0.03801 | 0.04149 | -0.00347 |

Stage-shuffle null:

- Permutations: 1000.
- Observed mean same-minus-cross statistic: 0.03429.
- Null mean: -0.00028.
- Null 95% interval: [-0.00545, 0.00536].
- Empirical p-value: 0.000999.
- Standardized effect size: 12.48.

Interpretation:

O1 supports the claim that controller-defined execution stages induce statistically non-random stage-conditioned memory-access locality. The aggregate effect is strong and passes the stage-shuffle null test. The strongest contribution comes from Plan-stage retrieval. Ground and Reason are positive but smaller; Reflect is slightly negative in same-minus-cross overlap.

Caveat:

Use the phrase `controller-defined execution stage`. Do not describe O1 labels as source-provided or benchmark-provided stages/phases. Do not claim that every individual stage has strong positive locality.

## 9. Figures

Generated figures:

- `figures/Fig_M1_history_scale_latency.png`
- `figures/Fig_M1_history_scale_traffic.png`
- `figures/Fig_M2_frontier_cost_evidence.png`
- `figures/Fig_M2_budget_oracle_overlap.png`
- `figures/Fig_M2_budget_task_quality.png`
- `figures/Fig_M3_tile_utilization.png`
- `figures/Fig_M3_access_amplification.png`
- `figures/Fig_M3_scatter_ratio_by_query_type.png`
- `figures/Fig_O1_same_vs_cross_stage_overlap.png`
- `figures/Fig_O1_stage_shuffle_null.png`
- `figures/Fig_O1_stage_pair_recurrence.png`

Recommended figure usage:

- Use M2 budget-oracle-overlap and M2 frontier-cost-evidence to show flat expansion tradeoff.
- Use M3 tile-utilization/access-amplification to show physical scattering/over-fetch headroom.
- Use O1 same-vs-cross and O1 stage-shuffle-null to show controller-defined stage locality.
- Be cautious with `Fig_M2_budget_task_quality.png`; answer quality is not a strong claim from this package.

## 10. Validation

Final validation passed:

```text
reports/VALIDATION.md: PASSED
```

Validated properties include:

- Required artifacts exist.
- M1/M2 retrieval rows satisfy schema requirements.
- O1 formal rows satisfy stage trace schema requirements.
- O1 has exactly four stages per query.
- Selected memories do not cross persona boundaries.
- Selected memories do not come from future sessions, with current-session context allowed by config.
- O1 stage-shuffle null has at least 1000 permutations in final validation.

## 11. Strongest Supported Claims

The following claims are supported by the current formal results:

1. Longitudinal medical histories and official distractors create large memory-state footprints for flat retrieval.
2. Expanding a flat retrieval frontier increases selection/reranking cost.
3. Larger frontier budgets improve reranker-oracle overlap, even though session-level Evidence Recall@5 is mostly saturated.
4. Chronological physical placement scatters jointly selected top memories across multiple tiles, creating over-fetch headroom.
5. A fixed auditable controller induces statistically non-random stage-conditioned memory-access locality, especially for planning-stage retrieval.

## 12. Claims to Avoid

Avoid these claims:

- Do not claim hypergraph superiority.
- Do not claim PIM speedup.
- Do not claim the final Method-1 layout is evaluated.
- Do not claim larger frontier budgets improve answer quality.
- Do not claim evidence completion failure means retrieval failure without explaining the session-level evidence anchor.
- Do not call O1 stages source-provided, benchmark-provided, or ground-truth cognitive phases.
- Do not claim every O1 stage independently has strong positive locality.
- Do not use the oracle tile-packing lower bound as if it were an implemented method.

## 13. Suggested GPT Analysis Prompts

Use one or more of the following prompts with this brief:

1. "Given this motivation experiment brief, propose a concise paper Motivation subsection that stays within the claim boundaries."
2. "Identify which figures should appear in the main paper and which should be appendix-only, and justify the choice."
3. "Rewrite the M1-M3-O1 result narrative as a single coherent evidence chain without claiming hypergraph or PIM superiority."
4. "Critique the weak points of this motivation package and suggest additional analyses that would strengthen it."
5. "Draft caption text for M2, M3, and O1 figures using only claims supported by the reported data."
6. "Propose a reviewer-facing explanation for why Evidence Completion@5 is zero and why this does not invalidate the motivation experiment."

## 14. Key Source Files for Follow-Up

Reports:

- `reports/FINAL_MOTIVATION_SUMMARY.md`
- `reports/M1_interpretation.md`
- `reports/M2_interpretation.md`
- `reports/M3_interpretation.md`
- `reports/O1_interpretation.md`
- `reports/00_pipeline_delta.md`
- `reports/VALIDATION.md`

Aggregates:

- `aggregate/M1_history_scale_summary.csv`
- `aggregate/M2_frontier_budget_summary.csv`
- `aggregate/M3_tile_scattering.csv`
- `aggregate/O1_same_cross_stage_stats.csv`
- `aggregate/O1_stage_shuffle_null.csv`
- `aggregate/O1_pair_recurrence.csv`

Configs:

- `configs/data_medmemorybench_en.yaml`
- `configs/retrieval_flat.yaml`
- `configs/model_qwen3_4b.yaml`
- `configs/stage_controller_v1.yaml`
- `configs/m1_history_scale.yaml`
- `configs/m2_budget.yaml`
- `configs/m3_tile_scattering.yaml`
- `configs/o1_stage_locality.yaml`
