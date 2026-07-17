# Final Motivation V4 Figure Guide

| Result | Figure | What was tested | Metric | Supports | Does not support | Placement |
|---|---|---|---|---|---|---|
| M1 | `Fig_M1_v4_history_footprint_traffic.png` | history scale/noise footprint | host/index/payload bytes | workload growth | universal latency law | appendix |
| M2 | `Fig_M2_v4_frontier_cost_reference_fidelity.png` | frontier budgets | latency and oracle overlap | flat expansion cost/reference fidelity | answer quality | main |
| M3 | `Fig_M3_v4_tile_access_amplification.png` | chronological tile placement | access amplification | physical over-fetch headroom | implemented layout superiority | main |
| O1 within-trace | `Fig_O1_v4_same_cross_stage_locality.png` | controller-defined stages | same/cross overlap | non-random access locality | source-provided phases | main |
| O1 held-out | `Fig_O1_v4_heldout_stage_transfer.png` | train-earlier/test-later transfer | stage advantage | future transfer if guardrails pass | universal stage locality | main only if pass |
| C strict evidence | `Fig_M2_v4_budget_cost_strict_keypoint_recall.png` | deterministic keypoint subset | strict recall@5 | stricter evidence behavior | human-verified gold | appendix unless high coverage |
