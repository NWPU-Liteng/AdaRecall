# GPT Analysis Brief: Method-1 Motivation Experiments on MedMemoryBench v4

Updated: 2026-07-06

This document is intended to be pasted into GPT or another analysis assistant. It summarizes the current MedMemoryBench motivation package after the v4 completion update: experimental setting, what changed from v3, new audits, new held-out transfer experiment, strict source-keypoint subset, formal results, figure roles, and claim boundaries.

## 1. Package Location and Purpose

V4 package:

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260705_method1_motivation_medmemorybench_v4_completion
```

V3 source package reused by v4:

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3
```

The v4 package extends v3 without modifying v3. It reuses unchanged v3 artifacts by symlink where appropriate and adds new v4 scripts, aggregates, manifests, figures, and reports under the v4 directory only.

Purpose:

Strengthen the Method-1 motivation/workload-characterization package. V4 still does not evaluate the final hypergraph layout, PIM runtime, replication policy, online remapping, or dense-vs-graph-vs-hypergraph method comparison.

## 2. Motivation Chain

The intended evidence chain is:

```text
Longitudinal personal-health histories accumulate many medically relevant,
revised, and distracting memory items.
        ->
A flat retriever must expand its candidate frontier to preserve reference-fidelity,
but expansion increases materialization and reranking cost.
        ->
The selected high-value memories are scattered across chronological memory tiles,
so tile-granular near-memory execution would over-fetch irrelevant content.
        ->
Under a fixed, auditable text-only controller, different controller-defined
execution stages access different but recurrent memory subsets.
        ->
V4 adds that some stage-conditioned co-access structure learned from earlier
histories transfers to later held-out accesses.
        ->
This motivates stage-conditioned structural memory organization.
```

Claim boundary:

The package motivates structural/stage-aware memory organization. It does not prove final method superiority, hypergraph superiority, PIM speedup, energy reduction, or answer-quality improvement.

## 3. Data and Workload

Dataset:

- Official English MedMemoryBench files are loaded locally from:

```text
/data2/user25263093/userliteng/26_code/preTest/data/external/MedMemoryBench
```

Schema/workload facts inherited from v3:

- English queries extracted: 1939.
- Personas: 20.
- Query types: 6.
- `dialogues_with_noise` is available.
- Memory granularity: chronological patient-doctor turn-pair memory item, plus one singleton.
- Total memory items: 42,034.
- Memory item distribution:
  - `medical`: 15,988.
  - `noise_family`: 13,026.
  - `noise_health`: 13,020.

Evidence mapping inherited from v3:

- Session-level evidence mapping coverage: 1,935/1,939 queries, 99.79%.
- Original evidence anchor:

```text
query source_key_points.session_id
    -> all medical turn-pair memories for that persona/session
```

This session-level anchor is conservative and coarse. It makes Evidence Recall@5 low and Evidence Completion@5 usually zero because the session-level gold set is often much larger than top-5.

## 4. Model and Runtime Settings

Immutable components:

- Controller/answer model: local Qwen3-4B path `../../../dir`.
- Embedding model: `../../models/bge-base-en-v1.5`.
- Reranker: `../../models/cross-encoder-ms-marco-MiniLM-L6-v2`.
- Retrieval style: flat retrieval with ANN overfetch, cross-encoder reranking, final top-k selection.
- Final selected evidence count: `K_o = 5`.
- Core high-budget frontier: `K_f = 64`.
- Controller stages:
  - Ground
  - Plan
  - Reason
  - Reflect

GPU/runtime:

- M1/M2 v3 formal runs used `CUDA_VISIBLE_DEVICES=3`.
- O1 v3 and B v4 GPU runs used `LOCAL_QWEN_CUDA_DEVICE=6`.
- In this managed environment, setting `CUDA_VISIBLE_DEVICES=<idx>` made `torch.cuda.is_available()` false. Leaving `CUDA_VISIBLE_DEVICES` unset and setting `LOCAL_QWEN_CUDA_DEVICE=6` worked.
- V4 runtime/device information is recorded in:

```text
manifests/runtime_manifest.json
```

## 5. What v4 Adds Over v3

V3 had already established:

1. Larger flat retrieval frontiers increase selection and reranking cost.
2. Larger frontiers increasingly match the high-budget reranker reference.
3. Chronological placement scatters selected memories across multiple tiles.
4. A fixed controller produces statistically non-random stage-conditioned access locality.

V4 adds:

1. A provenance/hash audit for the local dataset files.
2. A strict causal-leakage audit over v3 formal M1/M2/O1 selected memories.
3. A candidate-path occupancy audit proving the M2 budget changes the actual reranked frontier.
4. A controller-overhead positioning report.
5. A new 240-query O1-v4 stage trace population.
6. A chronological train-earlier/test-later held-out stage-locality transfer experiment.
7. A deterministic strict source-keypoint-aligned evidence subset.
8. A future human evidence-review packet.
9. V4 final figures, summary, claim boundaries, figure guide, and validation.

## 6. V3 Reuse

The v4 package reuses unchanged v3 artifacts by symlink. Reuse is documented in:

```text
reports/V3_REUSE_MANIFEST.md
```

Key reused artifacts include:

- v3 M1/M2/M3/O1 formal aggregates.
- v3 memory items and query JSONL files.
- v3 schema and config files.
- v3 evidence mapping files.

Copied-for-local-modification scripts include:

- `scripts/medmmb_common.py`
- `scripts/run_o1_stage_trace.py`
- `scripts/compute_o1_null_models.py`
- `scripts/monitor_medmmb_v4_progress.py`

## 7. Work Package A: Audits

### A1. Dataset Provenance Audit

Artifacts:

- `reports/DATASET_PROVENANCE_AUDIT.md`
- `manifests/dataset_file_hashes.sha256`

Key points:

- SHA256 hashes were generated for every local source data file used.
- Dataset location is recorded.
- Official repository revision could not be recovered from the local dataset copy.
- Exact Hugging Face revision and authoritative download date could not be verified from local files.
- License could not be verified unless present in local documentation.

### A2. Causal Leakage Audit

Artifacts:

- `aggregate/A2_causal_leakage_audit.jsonl`
- `reports/A2_CAUSAL_LEAKAGE_AUDIT.md`

Checked selected memories from v3 formal M1, M2, and O1.

Results:

- Total checked selected memories: 15,640.
- Future-session violations: 0.
- Same-session future-turn violations: 0.
- Cross-persona violations: 0.
- Missing timestamp/session metadata cases: 0.
- Missing same-session turn metadata cases: 327.

Interpretation:

No future-session, same-session future-turn, or cross-persona selected-memory violation was found. The current formal results are not invalidated by this audit. Some same-session turn anchors are not exposed in exported query rows, so those cases are reported separately instead of silently assumed.

### A3. Candidate-Path Occupancy Audit

Artifacts:

- `aggregate/A3_candidate_path_occupancy.csv`
- `reports/A3_CANDIDATE_PATH_AUDIT.md`

Main finding:

M2 budget changes the real reranked frontier, not just a config label.

For frontier budgets:

- budget 8 -> `n_reranked = 8`, frontier fill ratio = 1.0.
- budget 16 -> `n_reranked = 16`, frontier fill ratio = 1.0.
- budget 32 -> `n_reranked = 32`, frontier fill ratio = 1.0.
- budget 64 -> `n_reranked = 64`, frontier fill ratio = 1.0.

For all budgets:

- `n_final_selected = 5`.
- selection fill ratio = 1.0.
- fraction `n_reranked < K_f` = 0.
- fraction `n_final_selected < K_o` = 0.

### A4. Controller-Overhead Positioning

Artifact:

- `reports/A4_CONTROLLER_OVERHEAD_POSITIONING.md`

Results from O1 formal traces:

- O1 formal rows: 480.
- Mean controller intent-generation latency: 7037.96 ms.
- Mean retrieval-selection latency after intent generation: 147.57 ms.
- Controller/retrieval latency ratio: 47.69x.

Required wording:

The Qwen3-4B controller is used only to reconstruct and instrument an offline stage-aware workload for motivation analysis. It is not an additional online component of the final runtime design.

The controller latency is reported explicitly and is not hidden or subtracted.

## 8. Work Package B: Held-Out Future Stage-Locality Transfer

This is the highest-priority v4 addition.

Goal:

Test whether stage-conditioned co-access structure learned from earlier histories remains useful for later histories.

Trace population:

- B formal trace rows: 960.
- Queries: 240.
- Stages per query: 4.
- Stages:
  - ground
  - plan
  - reason
  - reflect
- Frontier budget: `K_f = 64`.
- Final selected count: `K_o = 5`.
- No answer generation.
- Trace artifact:

```text
aggregate/B_stage_trace_v4.jsonl
```

Trace quality:

- Parsed intent rate: 941/960 = 98.02%.
- All rows have `n_reranked = 64`.
- Stage-level mean controller intent latency:
  - ground: 6909.49 ms.
  - plan: 6749.55 ms.
  - reason: 6811.46 ms.
  - reflect: 6757.42 ms.
- Stage-level mean selection latency:
  - ground: 231.67 ms.
  - plan: 232.40 ms.
  - reason: 236.38 ms.
  - reflect: 232.88 ms.

Split:

- Per persona chronological split.
- Earliest 60% of valid query anchors -> train period.
- Latest 40% -> held-out test period.
- Split manifest:

```text
aggregate/B_chronological_split_manifest.csv
```

Transfer metric:

Train-period stage-conditioned co-access graphs are built from selected memory IDs. For each held-out test invocation, compute:

- seen-node coverage;
- stage cohesion;
- global cohesion;
- cross-stage cohesion;
- stage advantage = stage cohesion - cross-stage cohesion.

Statistics:

- Query-level bootstrap confidence intervals: 1000 samples.
- Stage-label shuffle null: 1000 permutations.

Artifacts:

- `aggregate/B_heldout_stage_transfer_per_query.csv`
- `aggregate/B_heldout_stage_transfer_summary.csv`
- `aggregate/B_heldout_stage_shuffle_null.csv`
- `reports/B_HELDOUT_STAGE_TRANSFER.md`

Held-out transfer results:

| Stage | Test invocations | Seen-node coverage | Stage advantage | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| ground | 106 | 0.449 | 0.0759 | [0.0380, 0.1194] |
| plan | 106 | 0.515 | 0.3700 | [0.2342, 0.5210] |
| reason | 106 | 0.425 | 0.0358 | [-0.0025, 0.0736] |
| reflect | 106 | 0.475 | 0.0752 | [0.0368, 0.1155] |
| aggregate | 424 | 0.466 | 0.1392 | [0.1012, 0.1816] |

Stage-shuffle null:

- Observed aggregate statistic: 0.1392.
- Null mean: -0.0004.
- Null 95% interval: [-0.0214, 0.0185].
- Empirical p-value: 0.0010.
- Standardized effect size: 13.6842.

Interpretation:

The held-out transfer result passes v4 guardrails:

- seen-node coverage is above 30%;
- aggregate stage advantage is positive;
- aggregate stage advantage is above the shuffled-stage null with p < 0.05.

Allowed claim:

Earlier controller-defined stage-conditioned access structure shows measurable transfer to later held-out accesses.

Important caveat:

Transfer is concentrated in planning-stage retrieval. Reason is weak because its bootstrap CI crosses zero. Do not claim universal strong transfer for every stage.

## 9. Work Package C: Strict Source-Keypoint-Aligned Evidence Subset

Goal:

Build a stricter deterministic subset aligned to official `source_key_points`, without pretending this is human-verified gold evidence.

Artifacts:

- `aggregate/C_keypoint_alignment_rows.jsonl`
- `aggregate/C_keypoint_alignment_summary.csv`
- `aggregate/C_strict_evidence_budget_summary.csv`
- `reports/C_KEYPOINT_ALIGNMENT_AUDIT.md`
- `reports/C_STRICT_EVIDENCE_INTERPRETATION.md`
- `artifacts/evidence_review_packet.csv`
- `reports/EVIDENCE_REVIEW_GUIDE.md`

Alignment rule:

A query enters the Strict Source-Keypoint-Aligned Subset only if all official source key points have at least one causally eligible high-confidence deterministic memory alignment.

High-confidence rule:

```text
normalized exact containment OR lexical similarity >= 0.95
```

Forbidden in alignment:

- gold answer text;
- reranker output;
- final answer output;
- LLM judge;
- manually invented evidence IDs.

Coverage:

- Formal M2 queries considered: 240.
- Strict Source-Keypoint-Aligned Subset queries: 8.
- Strict subset coverage: 8/240 = 0.0333.

Because coverage is only 3.33%, this result should be appendix material.

Budget summary on strict subset:

| Frontier budget | Strict subset queries | StrictKeypointRecall@5 | StrictKeypointCompletion@5 | RerankerOracleOverlap@5 | Selection latency ms | Rerank latency ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 0.750 | 0.750 | 0.650 | 94.15 | 57.57 |
| 16 | 8 | 0.625 | 0.625 | 0.875 | 93.85 | 81.39 |
| 32 | 8 | 0.750 | 0.750 | 0.950 | 147.13 | 135.17 |
| 64 | 8 | 0.750 | 0.750 | 1.000 | 229.79 | 217.59 |

Interpretation:

Strict recall is approximately flat across budgets on this very small deterministic subset, while reranker reference fidelity still improves. Because coverage is low, use this only as an appendix robustness check.

Do not call this subset human-verified evidence ground truth.

## 10. V3 Baseline Results Still Used in V4

### M1: History-Scale Profiling

Formal artifacts:

- `aggregate/M1_history_scale_invocations.jsonl`: 1688 rows.
- `aggregate/M1_history_scale_summary.csv`: 9 rows.
- Formal query coverage: 240/240.

Key observations:

- Medical-only full-history mean selection latency: 132.29 ms.
- Medical-plus-official-noise full-history mean selection latency: 134.25 ms.
- Frontier fill ratio stayed at 1.0 across reported scales.
- Host/index/payload byte footprint grows with larger history scales and official noise.

Interpretation:

M1 supports workload growth/footprint pressure. It should be appendix or supporting material because latency is not universally monotonic at every scale.

### M2: Frontier Budget Study

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

M2 supports flat frontier cost and high-budget reranker-reference fidelity. It does not support answer-quality improvement with larger frontier budgets.

### M3: Chronological Tile Scattering

Formal artifact:

- `aggregate/M3_tile_scattering.csv`: 2160 rows.
- Formal query coverage: 240/240.
- Source selections: fresh M2 formal budget-64 top-5 rows.

Mean results:

| Tile capacity | Layout | Activated tiles | Access amplification | Scatter ratio |
|---:|---|---:|---:|---:|
| 8 | chronological | 4.50 | 7.21 | 4.50 |
| 8 | random_per_persona | 4.94 | 7.90 | 4.94 |
| 8 | oracle tile-packing lower bound | 1.00 | 1.60 | 1.00 |
| 16 | chronological | 4.21 | 13.48 | 4.21 |
| 16 | random_per_persona | 4.90 | 15.68 | 4.90 |
| 16 | oracle tile-packing lower bound | 1.00 | 3.20 | 1.00 |
| 32 | chronological | 3.84 | 24.56 | 3.84 |
| 32 | random_per_persona | 4.85 | 31.04 | 4.85 |
| 32 | oracle tile-packing lower bound | 1.00 | 6.40 | 1.00 |

Interpretation:

M3 supports physical over-fetch headroom under chronological placement. It does not prove any implemented final layout.

### O1 Within-Trace Stage Locality

Formal artifacts:

- `aggregate/O1_formal_stage_trace.jsonl`: 480 rows.
- `aggregate/O1_same_cross_stage_stats.csv`: 4 rows.
- `aggregate/O1_stage_shuffle_null.csv`: 1000 permutations.
- Formal query coverage: 120/120.

Controller quality:

- Parsed intent JSON: 479/480 rows, 99.79%.
- Retrieval-goal prefix validity: 100%.
- Queries with more than one unique selected set across four stages: 120/120.

Same-stage versus cross-stage overlap:

| Stage | Same-stage overlap | Cross-stage overlap | Same-minus-cross |
|---|---:|---:|---:|
| ground | 0.06099 | 0.04708 | 0.01391 |
| plan | 0.16155 | 0.04155 | 0.11999 |
| reason | 0.05127 | 0.04453 | 0.00675 |
| reflect | 0.03801 | 0.04149 | -0.00347 |

Stage-shuffle null:

- Observed statistic: 0.03429.
- Null mean: -0.00028.
- Null 95% interval: [-0.00545, 0.00536].
- Empirical p-value: 0.000999.
- Standardized effect size: 12.48.

Interpretation:

O1 supports non-random access locality under fixed controller-defined execution stages. The strongest contribution is Plan-stage retrieval.

## 11. V4 Figure Set and Recommended Placement

Main-paper candidates:

1. `figures/Fig_M2_v4_frontier_cost_reference_fidelity.png`
   - Message: flat frontier expansion sharply raises selection/reranking cost while improving high-budget reranker-reference fidelity.

2. `figures/Fig_M3_v4_tile_access_amplification.png`
   - Message: chronological placement physically scatters high-value memories and creates tile-level over-fetch.

3. `figures/Fig_O1_v4_same_cross_stage_locality.png`
   - Message: controller-defined stages induce non-random access locality.

4. `figures/Fig_O1_v4_heldout_stage_transfer.png`
   - Message: earlier stage-conditioned co-access structure transfers to later held-out accesses.

Appendix candidates:

- `figures/Fig_M1_v4_history_footprint_traffic.png`
- `figures/Fig_M2_v4_budget_cost_strict_keypoint_recall.png`
- `figures/Fig_O1_v4_heldout_stage_shuffle_null.png`
- `figures/Fig_O1_v4_stage_seen_node_coverage.png`
- M2 answer-quality or session-level Evidence Completion@5 plots, if discussed at all.

Figure guide:

```text
reports/FINAL_MOTIVATION_V4_FIGURE_GUIDE.md
```

## 12. Validation

Final v4 validation:

```text
reports/VALIDATION_V4.md: PASSED
```

Validation confirms:

- v3 reuse manifest exists;
- v4 manifests and dataset hashes exist;
- causal leakage audit exists;
- candidate path occupancy audit exists;
- controller overhead report exists;
- B stage trace and held-out transfer aggregates exist;
- C strict evidence subset and review packet exist;
- all required v4 figures and final reports exist;
- held-out split is chronological per persona;
- stage-shuffle null uses at least 1000 permutations;
- bootstrap uses at least 1000 resamples;
- strict evidence subset uses deterministic alignment;
- M2 budgets correspond to real reranked frontier sizes.

## 13. Strongest Supported Claims After V4

The following claims are supported:

1. Flat frontier expansion increases selection and reranking cost.
2. Larger frontiers improve agreement with a high-budget reranking reference.
3. Chronological placement scatters selected memories across tiles and creates physical over-fetch headroom.
4. Under a fixed auditable controller, controller-defined execution stages exhibit non-random access locality.
5. Earlier controller-defined stage-conditioned access structure shows measurable transfer to later held-out accesses.

## 14. Claims to Avoid

Avoid these claims:

- Do not claim hypergraph superiority.
- Do not claim PIM speedup or energy reduction.
- Do not claim the final layout is implemented.
- Do not claim larger frontier budgets improve answer quality.
- Do not call controller stages source-provided or benchmark-provided cognitive phases.
- Do not present oracle tile packing as an implemented baseline.
- Do not call automatic keypoint alignment human-verified gold evidence.
- Do not overstate C strict evidence because strict subset coverage is only 3.33%.
- Do not claim every stage transfers equally; transfer is concentrated in Plan-stage retrieval and Reason is weak.

## 15. Suggested GPT Analysis Prompts

Use one or more of these prompts:

1. "Given this v4 motivation brief, write a concise paper Motivation subsection that stays within the claim boundaries."
2. "Choose the main-paper figures and appendix figures, and justify the placement."
3. "Rewrite the M2-M3-O1-B evidence chain as a coherent narrative without claiming hypergraph or PIM superiority."
4. "Critique the weak points of this package, especially strict evidence coverage and controller overhead."
5. "Draft figure captions for the four main-paper candidate figures using only supported claims."
6. "Explain how to present the held-out stage-transfer result without implying universal cognitive phases."
7. "Write a reviewer response explaining why the strict source-keypoint subset is appendix-only and why the session-level evidence anchor is conservative."

## 16. Key Files for Follow-Up

Reports:

- `reports/FINAL_MOTIVATION_V4_SUMMARY.md`
- `reports/FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md`
- `reports/FINAL_MOTIVATION_V4_FIGURE_GUIDE.md`
- `reports/VALIDATION_V4.md`
- `reports/A2_CAUSAL_LEAKAGE_AUDIT.md`
- `reports/A3_CANDIDATE_PATH_AUDIT.md`
- `reports/A4_CONTROLLER_OVERHEAD_POSITIONING.md`
- `reports/B_HELDOUT_STAGE_TRANSFER.md`
- `reports/C_KEYPOINT_ALIGNMENT_AUDIT.md`
- `reports/C_STRICT_EVIDENCE_INTERPRETATION.md`
- `reports/EVIDENCE_REVIEW_GUIDE.md`

Aggregates:

- `aggregate/A2_causal_leakage_audit.jsonl`
- `aggregate/A3_candidate_path_occupancy.csv`
- `aggregate/B_stage_trace_v4.jsonl`
- `aggregate/B_chronological_split_manifest.csv`
- `aggregate/B_heldout_stage_transfer_per_query.csv`
- `aggregate/B_heldout_stage_transfer_summary.csv`
- `aggregate/B_heldout_stage_shuffle_null.csv`
- `aggregate/C_keypoint_alignment_rows.jsonl`
- `aggregate/C_keypoint_alignment_summary.csv`
- `aggregate/C_strict_evidence_budget_summary.csv`

Figures:

- `figures/Fig_M2_v4_frontier_cost_reference_fidelity.png`
- `figures/Fig_M3_v4_tile_access_amplification.png`
- `figures/Fig_O1_v4_same_cross_stage_locality.png`
- `figures/Fig_O1_v4_heldout_stage_transfer.png`
- `figures/Fig_M1_v4_history_footprint_traffic.png`
- `figures/Fig_M2_v4_budget_cost_strict_keypoint_recall.png`
- `figures/Fig_O1_v4_heldout_stage_shuffle_null.png`
- `figures/Fig_O1_v4_stage_seen_node_coverage.png`
