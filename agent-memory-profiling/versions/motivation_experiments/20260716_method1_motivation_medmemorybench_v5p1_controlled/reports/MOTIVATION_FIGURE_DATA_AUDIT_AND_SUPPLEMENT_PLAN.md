# Motivation Figure/Data Audit and Supplement Plan

## Decision

The previous five-panel layout mixed workload counts, reference agreement,
physical bytes, and timing in incompatible visual units. The revised set uses
one scientific question per figure, one axis system, query-level uncertainty,
and no explanatory prose inside plotted lines.

## Current evidence audit

| Experiment | Current data status | Main-figure decision | Additional data required |
| --- | --- | --- | --- |
| M1 history pressure | 486 controlled rows, same 81 queries in every cell, three scales | Use a stacked component bar of traffic normalized to the 128-memory baseline; absolute MB is audit-only | Main figure needs no new run. For a stronger generality claim, repeat the same protocol on a second corpus or add a larger-history cohort; the current MedMemoryBench medical-only histories cannot provide a common 1024 cell |
| S1 evidence structure | 1,355 query-level rows with group counts and B=4/8/16/32 coverage | Use grouped bars of strict all-group completion by required group count at B=8 and B=32, with binomial/bootstrap intervals | No new run; recompute intervals from the existing per-query file |
| M2 flat frontier | 240 queries x B=8/16/32/64; latency and B=64 reference overlap exist; gold memory recall is nearly flat and completion is zero | Use a cost-vs-reference-agreement scatter, explicitly labelled as a reranker-reference diagnostic | For a quality claim, add a task-level or group-level correctness metric; do not infer answer quality from the current reference overlap |
| M3 Tile scattering | 240 query selections, layouts, capacities 8/16/32, target-k fixed at 5 | Use grouped bars of access amplification over Tile capacity | Target-k 5/10/20 has now been derived from the saved B=64 reranked lists; add a compact-placement baseline only if making a stronger placement-generalization claim |
| S2 content maintenance | Four growth points, one deterministic insertion trajectory, 128 probes, exact lookup check | Appendix-only grouped ratio bars: full rebuild work/time divided by incremental work/time | Repeat at least 5 seeds and two insertion mixes if timing generality is needed; never use it as evidence for dynamic agent access or hypergraph causality |

## Priority order

1. Re-aggregate M2 gold/reference metrics, S1 confidence intervals, and M3
   confidence intervals from existing query-level logs. This is a no-GPU data
   completion step.
2. Replace the five old composite images with the standalone scripts in
   `runs/v5p1_figures_revised/`. The recommended main figures are M1 traffic,
   S1 group completion, M2 cost-reference tradeoff, and M3 Tile over-fetch.
   S2 remains appendix-only.
3. The M3 target-k sensitivity supplement (`k=5,10,20`) is now available as a
   deterministic post-processing result; add a compact-placement baseline only
   if the paper makes a placement-generalization statement.
4. Run the M1 cross-corpus or larger-history supplement only if the paper uses
   a claim stronger than “within this workload, retained history scale
   amplifies traffic.”
5. Repeat S2 only when reporting timing as a general systems result; otherwise
   keep the existing deterministic operation-count evidence in the appendix.

## Claim boundaries after revision

The revised figures support: multi-group evidence demand, cost versus a clearly
defined flat-reranker reference, logical-Tile over-fetch, and normalized
history-induced traffic amplification. They do not support formal-method
superiority, agent answer-quality improvement, stage-aware retrieval,
access-order learning, dynamic hypergraph updates, or PIM hardware speedup.
