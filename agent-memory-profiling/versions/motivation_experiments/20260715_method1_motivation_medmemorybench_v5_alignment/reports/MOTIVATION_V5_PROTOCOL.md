# Motivation v5 Protocol

## Purpose

The protocol establishes method-independent workload facts that motivate the
problem addressed by Method 1. It must not compare the formal v21 method with a
baseline or use v21 predictions as evidence.

## Reused evidence

- M1: history scale, candidate-processing cost, and host traffic.
- M2: frontier budget, selection/reranking cost, and evidence/reference tradeoff.
- M3: chronological/random placement, activated Tiles, utilization, and access
  amplification. Oracle packing is a lower bound only.

The v4 O1 controller-stage results remain an appendix workload observation. They
are excluded from the main motivation chain because stage labels are not v21
inputs.

## S1: Evidence structure

Inputs are raw queries/memories, train-only evidence groups, and frozen query and
memory embeddings. For each query, the analysis records group count, acceptable
alternative count, cross-group overlap, source-session span, and flat cosine
coverage at B in {4, 8, 16, 32}. The flat reference is a workload probe, not a
method comparator.

The expected result is that multi-group queries are common, groups have multiple
acceptable alternatives, and a flat top-B set often leaves at least one group
uncovered or spends capacity on redundant members.

## S2: Generic content maintenance

Memory items are sorted by chronological metadata only to construct deterministic
80% base plus 1/5/10/20% append slices. A plain token-to-memory inverted index is
rebuilt from scratch and updated append-only. We record update operations, changed
postings, index rebuild work, wall time, lookup exactness, and logical Tile
activation for probe lookups. Ordering is not interpreted as agent access order.

The expected result is that append-only maintenance changes only a small fraction
of postings at each growth point while full rebuild work scales with the entire
corpus; lookup results must match exactly.
