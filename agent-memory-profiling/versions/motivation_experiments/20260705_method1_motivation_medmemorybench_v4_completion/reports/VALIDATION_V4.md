# Validation V4

- [x] `reports/V3_REUSE_MANIFEST.md` exists
- [x] `reports/DATASET_PROVENANCE_AUDIT.md` exists
- [x] `manifests/dataset_file_hashes.sha256` exists
- [x] `manifests/runtime_manifest.json` exists
- [x] `aggregate/A2_causal_leakage_audit.jsonl` exists
- [x] `reports/A2_CAUSAL_LEAKAGE_AUDIT.md` exists
- [x] `aggregate/A3_candidate_path_occupancy.csv` exists
- [x] `reports/A3_CANDIDATE_PATH_AUDIT.md` exists
- [x] `reports/A4_CONTROLLER_OVERHEAD_POSITIONING.md` exists
- [x] `aggregate/B_stage_trace_v4.jsonl` exists
- [x] `aggregate/B_chronological_split_manifest.csv` exists
- [x] `aggregate/B_heldout_stage_transfer_per_query.csv` exists
- [x] `aggregate/B_heldout_stage_transfer_summary.csv` exists
- [x] `aggregate/B_heldout_stage_shuffle_null.csv` exists
- [x] `reports/B_HELDOUT_STAGE_TRANSFER.md` exists
- [x] `aggregate/C_keypoint_alignment_rows.jsonl` exists
- [x] `aggregate/C_keypoint_alignment_summary.csv` exists
- [x] `aggregate/C_strict_evidence_budget_summary.csv` exists
- [x] `reports/C_KEYPOINT_ALIGNMENT_AUDIT.md` exists
- [x] `reports/C_STRICT_EVIDENCE_INTERPRETATION.md` exists
- [x] `artifacts/evidence_review_packet.csv` exists
- [x] `reports/EVIDENCE_REVIEW_GUIDE.md` exists
- [x] `figures/Fig_M1_v4_history_footprint_traffic.png` exists
- [x] `figures/Fig_M2_v4_frontier_cost_reference_fidelity.png` exists
- [x] `figures/Fig_M3_v4_tile_access_amplification.png` exists
- [x] `figures/Fig_O1_v4_same_cross_stage_locality.png` exists
- [x] `figures/Fig_O1_v4_heldout_stage_transfer.png` exists
- [x] `figures/Fig_O1_v4_heldout_stage_shuffle_null.png` exists
- [x] `figures/Fig_O1_v4_stage_seen_node_coverage.png` exists
- [x] `figures/Fig_M2_v4_budget_cost_strict_keypoint_recall.png` exists
- [x] `scripts/figure_plots/plot_fig_m1_history_footprint_traffic.py` exists
- [x] `scripts/figure_plots/plot_fig_m2_frontier_cost_reference_fidelity.py` exists
- [x] `scripts/figure_plots/plot_fig_m3_tile_access_amplification.py` exists
- [x] `scripts/figure_plots/plot_fig_o1_same_cross_stage_locality.py` exists
- [x] `scripts/figure_plots/plot_fig_o1_heldout_stage_transfer.py` exists
- [x] `scripts/figure_plots/plot_fig_o1_heldout_stage_shuffle_null.py` exists
- [x] `scripts/figure_plots/plot_fig_o1_stage_seen_node_coverage.py` exists
- [x] `scripts/figure_plots/plot_fig_m2_budget_cost_strict_keypoint_recall.py` exists
- [x] `scripts/figure_plots/run_all_figures.py` exists
- [x] `reports/FIGURE_DATA_SOURCES.md` exists
- [x] `reports/FINAL_MOTIVATION_V4_SUMMARY.md` exists
- [x] `reports/FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md` exists
- [x] `reports/FINAL_MOTIVATION_V4_FIGURE_GUIDE.md` exists
- [x] held-out split is chronological per persona
- [x] stage-shuffle null uses >= 1000 permutations
- [x] bootstrap uses >= 1000 resamples
- [x] strict evidence subset uses deterministic alignment
- [x] M2 budgets correspond to real reranked frontier sizes

## Status Table

| Component | Status | Main-paper usable? | Claim |
|---|---|---|---|
| M1 | pass | appendix | history footprint/traffic |
| M2 | pass | yes | frontier cost/reference fidelity |
| M3 | pass | yes | tile scattering/over-fetch |
| O1 within-trace | pass | yes | non-random stage locality |
| O1 held-out transfer | see B report | conditional | earlier-to-later transfer |
| strict evidence subset | see C report | appendix | deterministic keypoint subset |

## Overall

PASSED

