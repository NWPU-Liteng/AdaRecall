# v5.1 Unified Data Dictionary

`aggregate/v5p1_motivation_final_data.csv` is long-form: one row is one
experiment/condition/metric aggregate. Confidence intervals are bootstrap 95%
intervals when the source summary provides them; blank intervals mean that the
source only supplied a deterministic aggregate.

| Field | Meaning |
| --- | --- |
| `experiment` | `M1_controlled_history_pressure`, `S1_evidence_structure`, `M2_flat_frontier_tradeoff`, `M3_tile_scattering`, `S2_generic_content_maintenance`, or `M1_feasibility_audit` |
| `data_role` | Controlled observation, reused observation, workload structure, reference baseline, appendix stress test, or audit |
| `query_type` | Benchmark query type; `all` is the aggregate across types |
| `condition` | M1 history condition, S1 evidence/reference role, M2 flat-history setting, M3 selected-memory setting, or S2 static index setting |
| `history_scale` | Retained memories for M1, or audit scale |
| `frontier_budget` | Flat frontier size for M1/M2/S1 |
| `tile_capacity` / `layout` | Logical Tile capacity and placement layout for M3/S2 |
| `growth_percent` | Appended corpus percentage for S2 |
| `metric` / `value` | Metric name and numeric value |
| `n` | Number of query/probe observations contributing to the aggregate |
| `common_query_count` | Frozen common query count where applicable; M1 is 81, M2/M3 are 240, S1 is 1,355, and S2 probes are 128 |
| `source_file` | Relative provenance path; all paths are validated to exist |
| `claim_role` | `main_motivation`, `supporting_motivation`, `appendix_only`, or `audit_only` |

M1 metrics include the primary dimensionless
`host_traffic_amplification_vs_128` ratio (paired per query against the
condition's 128-memory baseline), selection/query-encoding/ANN/reranking
latency, and secondary absolute host, metadata, payload, and candidate-vector
bytes; reranked count; and frontier fill ratio. M2 includes selection cost, host bytes, and overlap with
the B=64 reranker reference. M3 includes activated Tiles, access
amplification, Tile utilization, and scatter ratio. S1 includes evidence-group
count, multi-group fraction, group coverage, all-group coverage, redundancy,
and acceptable-memory ambiguity. S2 includes posting work, timing, exact
lookup mismatches, and probe Tile counts.
