# Motivation v5.1 Results

## Controlled M1 replacement

The old M1 trend is retired because its cells used different query sets and
different sample counts (`202/161/88` for 128/256/512 in one condition), and
its `full` endpoint was not a continuation of the bounded scales. The v5.1
audit starts with the same 240 balanced M2 workload queries, computes eligible
history and gold-evidence feasibility under both conditions, and freezes one
query set before retrieval is run.

The frozen cohort contains 81 queries: entity exact match 16, inference 16,
multi-hop 13, multiple choice 15, state update 5, and temporal localization
16. Every condition-scale cell contains exactly these 81 IDs. The selected
history scales are 128, 256, and 512; frontier budget is 32 and final top-k is
5. Histories are nested and retain the gold evidence deterministically.

The primary M1 quantity is dimensionless host-traffic amplification relative to
the same condition's 128-memory baseline. At 128 -> 512 retained memories it
is 3.18x for `medical_only` and 3.24x for `medical_plus_official_noise`. This
normalization removes dependence on embedding dimensionality, payload
serialization, and absolute corpus size. Absolute bytes are retained only as
an audit metric. Mean selection latency is 120.4 -> 139.9 ms and 110.1 ->
135.3 ms, respectively. The latency curve is not claimed to be strictly
monotone; the robust M1 signal is the controlled growth of normalized host-side
traffic. The formal run used the CUDA cross-encoder and is recorded in the
manifest.

## Reused and supplementary evidence

- S1: 1,355 train queries; 25.54% require more than one evidence group. For
  multi-hop queries, mean group count is 2.69 and all-groups-covered at B=8 is
  5.17%.
- M2: across the fixed 240-query workload, selection latency rises from 58.55
  ms at B=8 to 229.23 ms at B=64, while the B=64 reranker-reference overlap
  changes only modestly (the reference is not a gold answer-quality label).
- M3: at Tile capacity 16, chronological and random placement activate 4.21
  and 4.90 Tiles on average versus the 1-Tile oracle lower bound; their access
  amplification is 13.48x and 15.68x versus 3.20x.
- S2 (appendix only): at 20% deterministic append growth, 9.69% of full
  postings change, lookup mismatches are zero, and measured full rebuild time is
  8.60x the incremental update time. This is a static index maintenance stress
  test, not an agent access-chain or dynamic-update experiment.

## Claim boundary

The package supports history traffic pressure, multi-group evidence demand,
flat-frontier cost/fidelity tradeoff, and Tile-granular over-fetch. It does not
support Method-1 superiority, answer-quality improvement, stage-aware
retrieval, access-order learning, dynamic hypergraph updates, or PIM hardware
latency/energy claims.

## Revised figure set

The recommended figures are the standalone files under `figures/revised/`.
M1 uses normalized component traffic rather than two near-overlapping condition
curves. S1 uses strict completion by group count; M2 uses latency versus the
explicitly named B=64 reranker reference; M3 uses Tile-capacity grouped bars
and a separate target-k sensitivity supplement; S2 uses dimensionless
rebuild/incremental ratios and remains appendix-only. Figure-ready aggregates
are under `aggregate/revised_figure_data/`.
