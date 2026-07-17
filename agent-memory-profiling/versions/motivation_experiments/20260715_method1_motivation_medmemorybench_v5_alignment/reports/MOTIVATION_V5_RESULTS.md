# Motivation v5 Results

## Closure decision

The revised motivation chain passes all protocol guardrails. It is a workload
characterization package and does not claim that the formal v21 method is better
than a baseline.

| chain step | result |
|---|---|
| M1 history pressure | Reused v4 formal data; mean host traffic spans 538,338 to 1,725,647 bytes across the medical-only history scales. |
| M2 flat frontier tradeoff | Reused v4 data; weighted selection latency rises from 58.55 ms at B=8 to 229.23 ms at B=64. |
| M3 Tile scattering | Reused v4 data; chronological and random placement stay above the oracle Tile-packing lower bound. |
| S1 evidence structure | 1,355 train queries; 25.54% have more than one evidence group. Multi-hop queries average 2.69 groups, and flat B=8 covers all groups for only 5.17% of them. |
| S2 content maintenance | At 20% appended content, changed postings are 9.69% of full postings; the current run measures full rebuild at 8.60x the incremental update time. All 128 probe lookups match exactly. |

## Main figures

- `figures/Fig_V5_M1_history_pressure.png`
- `figures/Fig_V5_M2_flat_frontier_tradeoff.png`
- `figures/Fig_V5_M3_tile_activation.png`
- `figures/Fig_V5_S1_evidence_structure.png`
- `figures/Fig_V5_S2_content_maintenance.png`

## Interpretation

The evidence supports a general workload problem: longer histories create
candidate and traffic pressure; queries can require several distinct evidence
groups with alternative memories; flat top-B selection often fails to cover all
groups; and useful items are physically scattered across Tiles. Under content
growth, a generic append-only index touches only new postings while preserving
exact lookup behavior, whereas a full rebuild processes the complete corpus.

These observations motivate structured, bounded, and incrementally maintainable
memory organization. They do not establish hypergraph superiority, stage-aware
retrieval, access-chain learning, causal dynamic updates, or PIM hardware gains.
