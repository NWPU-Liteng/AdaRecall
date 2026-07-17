# Figure Data Sources

The split plotting scripts live under:

```text
scripts/figure_plots/
```

Each plotting script is standalone and contains the plotted data explicitly as Python lists, dictionaries, or histogram bin counts. The original aggregate files remain the reproducible source of record.

| Figure | Plot script | Hardcoded data origin | Original aggregate location |
|---|---|---|---|
| `Fig_M1_v4_history_footprint_traffic.png` | `scripts/figure_plots/plot_fig_m1_history_footprint_traffic.py` | `history_condition`, `history_scale`, `host_memory_bytes_mean` rows | `aggregate/M1_history_scale_summary.csv` -> `../20260704_method1_motivation_medmemorybench_v3/aggregate/M1_history_scale_summary.csv` |
| `Fig_M2_v4_frontier_cost_reference_fidelity.png` | `scripts/figure_plots/plot_fig_m2_frontier_cost_reference_fidelity.py` | grouped means by `frontier_budget`: selection latency, rerank latency, oracle overlap@5 | `aggregate/M2_frontier_budget_summary.csv` -> `../20260704_method1_motivation_medmemorybench_v3/aggregate/M2_frontier_budget_summary.csv` |
| `Fig_M3_v4_tile_access_amplification.png` | `scripts/figure_plots/plot_fig_m3_tile_access_amplification.py` | grouped means by `layout` and `tile_capacity`: access amplification | `aggregate/M3_tile_scattering.csv` -> `../20260704_method1_motivation_medmemorybench_v3/aggregate/M3_tile_scattering.csv` |
| `Fig_O1_v4_same_cross_stage_locality.png` | `scripts/figure_plots/plot_fig_o1_same_cross_stage_locality.py` | same-stage and cross-stage overlap by controller stage | `aggregate/O1_same_cross_stage_stats.csv` -> `../20260704_method1_motivation_medmemorybench_v3/aggregate/O1_same_cross_stage_stats.csv` |
| `Fig_O1_v4_heldout_stage_transfer.png` | `scripts/figure_plots/plot_fig_o1_heldout_stage_transfer.py` | held-out stage advantage means and 95% bootstrap intervals | `aggregate/B_heldout_stage_transfer_summary.csv` |
| `Fig_O1_v4_heldout_stage_shuffle_null.png` | `scripts/figure_plots/plot_fig_o1_heldout_stage_shuffle_null.py` | explicit 40-bin histogram counts from 1,000 stage-shuffle statistics plus observed statistic | `aggregate/B_heldout_stage_shuffle_null.csv` |
| `Fig_O1_v4_stage_seen_node_coverage.png` | `scripts/figure_plots/plot_fig_o1_stage_seen_node_coverage.py` | seen-node coverage means by stage plus 30% guardrail | `aggregate/B_heldout_stage_transfer_summary.csv` |
| `Fig_M2_v4_budget_cost_strict_keypoint_recall.png` | `scripts/figure_plots/plot_fig_m2_budget_cost_strict_keypoint_recall.py` | strict subset selection latency and StrictKeypointRecall@5 by frontier budget | `aggregate/C_strict_evidence_budget_summary.csv` |

Run all figures:

```bash
conda run -n agent-memory-profiling python scripts/figure_plots/run_all_figures.py
```

Run one figure:

```bash
conda run -n agent-memory-profiling python scripts/figure_plots/plot_fig_m2_frontier_cost_reference_fidelity.py
```

The current aggregate data are under:

```text
aggregate/
```

M1/M2/M3/O1 within-trace aggregates in v4 are symlinks to the v3 package. B held-out transfer and C strict keypoint evidence aggregates were generated directly in v4.

