# Final Motivation V4 Summary

V4 strengthens v3 with provenance/causality audits, held-out stage-transfer testing, and a deterministic strict source-keypoint-aligned subset.

## Component Summary

| Component | Status | Main-paper usable? | Claim |
|---|---|---|---|
| M1 | pass | appendix | history footprint/traffic increases with scale and noise; latency is not universally monotonic |
| M2 | pass | yes | flat frontier expansion increases cost and improves high-budget reranker-reference fidelity |
| M3 | pass | yes | chronological placement scatters selected memories and creates over-fetch headroom |
| O1 within-trace | pass | yes | fixed controller-defined stages exhibit non-random access locality |
| O1 held-out transfer | pass | yes | earlier stage-conditioned access structure transfers to later held-out accesses |
| strict evidence subset | partial | appendix | deterministic strict subset coverage=0.033 |

See `reports/FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md` for allowed and forbidden claims.
