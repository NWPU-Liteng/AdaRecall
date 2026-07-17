# Motivation v5 Figure Guide

The five figures are method-independent workload characterizations. They do not
compare against the formal v21 method.

| Figure | Panels | Main message | Suggested placement |
|---|---|---|---|
| M1 history pressure | selection latency with 95% CI; host bytes with 95% CI; full-history stage breakdown | longer histories increase host work, while reranking dominates selection latency | main or early appendix |
| M2 flat frontier tradeoff | six query-type Pareto panels with latency and fidelity confidence intervals | larger flat frontiers approach the B=64 reranker reference only by paying rapidly increasing selection cost | main |
| M3 Tile activation | activated-Tile heatmap; access amplification; useful-payload utilization; query-level ECDF | naive placement scatters selected memories and creates Tile-granular over-fetch | main |
| S1 evidence structure | two query-type/budget heatmaps; evidence-group distribution; composition-vs-coverage curve | multi-group evidence demand is common, and flat top-B coverage deteriorates as group count grows | main |
| S2 content maintenance | deterministic posting work; log-scale timing; rebuild ratios; exactness/scope summary | local append maintenance avoids full rebuild work while preserving exact lookups | appendix unless content-maintenance dynamics are a central claim |

The intended main-text order is M1 -> S1 -> M2 -> M3. S2 closes the static
content-maintenance motivation but must not be described as an Agent access-stream
or access-chain result.
