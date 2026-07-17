# RESULTS

## Environment
- Server / GPU: validated on July 3, 2026 against the `agent-memory-profiling` Conda environment; current server exposes NVIDIA GeForce RTX 3090 GPUs.
- Conda environment: `agent-memory-profiling`.
- Git commit: unavailable because the workspace contains an empty `.git/` directory rather than a valid git metadata tree.
- Historical source runs: `20260616_phase_aware_real_agent_v3_formal` for M1/M2/M3 and `20260621_real_trace_adaptive_phase_v4_formal` for O1.

## M1: Memory-scale profiling
- Main observation: at scale 395934, budget 100, mean rerank latency is 654.7 ms and mean total latency is 3755.7 ms.
- Figure: `fig_m1_latency_breakdown` and `fig_m1_budget_vs_downstream_cost`.
- Caveat: this consolidated package reuses historical budgets `{50,100}` instead of a freshly rerun `{8,16,32,64}` grid.

## M2: Budget-quality trade-off
- Main observation: multi-evidence downstream selection cost rises from 422.6 ms to 870.1 ms as frontier budget increases from 50 to 100.
- Figure: `fig_m2_quality_cost_pareto`.
- Caveat: quality is proxied by gold-group hit and high-budget stability on the historical V3 synthetic-query workload, not by a benchmark-native answer metric.

## M2 Supplement: Fresh V4 Official-QA Budget Grid
- Real-GPU reruns completed on July 3, 2026 for `locomo` and `longmemeval_s` over guide-aligned budgets `{8,16,32,64}` with `shared_once`, `selection_only=false`, and `final_top_k=5`.
- The final rerun path uses metadata-aware answer prompting, deterministic short-answer contraction, and scorer-side handling for explicit alternative references such as `7 days. 8 days (including the last day) is also acceptable.` plus long-form insufficiency labels.
- Aggregate table: `aggregate/v4_official_qa_budget_grid_summary.csv` and `aggregate/v4_official_qa_budget_grid_summary.md`.
- Main cost observation on `locomo`: mean selection-path latency rises from `201.3` ms at budget 8 to `990.9` ms at budget 64; mean total latency stays in the `4.9-5.8` s band because answer generation dominates the wall clock.
- Main cost observation on `longmemeval_s`: mean selection-path latency rises from `87.3` ms at budget 8 to `466.7` ms at budget 64; mean total latency rises from `4.91` s to `5.38` s over the same sweep.
- Quality observation on `locomo`: formal `answer_exact_match` is now non-zero but low at `{0.0313, 0.0469, 0.0156, 0.0313}` for budgets `{8,16,32,64}`; `answer_token_f1` stays in the `0.19-0.23` range. This closes the previous "all-zero metric" failure mode, but the task score remains weak and non-monotonic.
- Quality observation on `longmemeval_s`: formal `answer_exact_match` is `{0.1563, 0.1406, 0.1406, 0.1250}` and `answer_token_f1` is `{0.2258, 0.2140, 0.2167, 0.1948}` for budgets `{8,16,32,64}`. The official QA path is therefore no longer degenerate, but it still does not yield a clean monotonic budget-quality curve.
- Retrieval-side signal: `avg_evidence_recall` stays roughly flat on `locomo` (`0.3333` to `0.3490`) and improves only modestly on `longmemeval_s` (`0.3906` to `0.4271`) as budget increases.
- Interpretation boundary: these fresh reruns now validate an end-to-end real-QA loop with non-zero benchmark-native task scores plus sharply budget-sensitive downstream cost, but they still do not support a strong claim that larger candidate budgets reliably improve final task quality.

## M2 Supplement: Auditable Query Strata on the Fresh V4 Grid
- Strata aggregate tables: `aggregate/v4_official_qa_query_strata.csv`, `aggregate/v4_official_qa_strata_population.csv`, and `aggregate/v4_official_qa_strata_summary.csv`.
- Strata construction is auditable and label-driven: `single_evidence` vs `multi_evidence` comes from `gold_turn_ids` count together with native `official_question_type`, and `long_range` is defined as the oldest-quartile subset by gold-memory position inside the visible history for each dataset.
- Formal subset sizes are usable but not huge: `locomo` contributes `26` single-evidence, `38` multi-evidence, and `13` long-range queries; `longmemeval_s` contributes `33`, `31`, and `13` respectively.
- `locomo` shows the clearest hard-subset gap: at budget 64, single-evidence queries reach mean `evidence_recall=0.5385`, while multi-evidence and long-range queries remain much lower at `0.2105` and `0.2308`; meanwhile downstream selection cost still rises from `334.0` ms to `1923.7` ms on multi-evidence and from `276.7` ms to `1767.0` ms on long-range.
- `locomo` stability also becomes budget-sensitive on the hard subsets: multi-evidence `high_budget_reranker_reference_stability` rises from `0.5316` at budget 8 to `1.0` at budget 64, and long-range rises from `0.5231` to `1.0`, which is consistent with a candidate-budget bottleneck even though official EM remains zero.
- `longmemeval_s` is harder to interpret from answer-side metrics, but the retrieval-side gap is still visible: multi-evidence stability rises from `0.3097` to `0.4516` as downstream cost rises from `119.2` ms to `775.8` ms, whereas single-evidence rises from `0.4424` to `0.5455`.
- A notable failure mode appears on `longmemeval_s` hard subsets: the mean selected-memory count is only `1.15` on long-range and `2.26` on multi-evidence even at budget 64, versus the configured `final_top_k=5`, so some questions never materialize a dense frontier to begin with.
- Interpretation boundary: these strata strengthen the claim that hard questions are retrieval-side sparse and budget-sensitive, but the answer-side strata remain noisy and non-monotonic. `longmemeval_s` does show non-zero task scores on some single-evidence and multi-evidence slices, yet the subset-level official EM is still too unstable to serve as a clean budget-scaling law.

## M3: Candidate scattering
- Main observation: chronological mean tile utilization at capacity 16 is 0.067, versus the ideal lower-bound utilization 0.495.
- Figure: `fig_m3_tile_scattering_utilization` and `fig_m3_tile_activation_vs_ideal`.
- Caveat: final evidence count follows the historical repo default `K_o=8`, not the guide-preferred `K_o=5`.

## O1: Phase-conditioned locality
- Main observation: real-trace adjacent overlap at window=1 is 0.7094; memory-shuffle drops it to 0.0843, while phase-shuffle stays at 0.7094 because it preserves the access sets and only perturbs labels.
- Figure: `fig_o1_phase_locality_matrix` and `fig_o1_null_model_gap`.
- Caveat: the available formal V4 traces cover `planning` and `reasoning` under `selection_only`, so phase coverage is narrower than the five-phase idealization in the guide.

## Readiness for Method Implementation
- What is now empirically supported: rerank-heavy selection cost exists in the historical real-GPU runs; the fresh V4 official-QA reruns confirm that downstream selection and rerank latency increase sharply with frontier budget on real QA traffic; the official QA loop now produces non-zero benchmark-native task scores instead of collapsing to all-zero EM; the auditable hard-query strata show that multi-evidence and long-range subsets are materially harder on the retrieval side than single-evidence subsets; selected memories are physically scattered under chronological placement; real traces show strong non-random adjacent overlap relative to memory-shuffled null traces.
- What remains unproven: a stable official task-score-vs-budget improvement curve on real QA, query-stratified M2 answer degradation on multi-evidence / long-range subsets, and any claim about hypergraph or PIM speedup.
