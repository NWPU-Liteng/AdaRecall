# Method 1 v21 Closure r2 Preregistration

This document freezes the remaining development protocol after completion of the sparse semantic rank index and retention of the r1 learned bundle-locality negative result. It does not unlock calibration or formal test.

## Primary Sparse Path

For each outer OOF unit, the primary path is: protected ANN seeds, QESA prototype prediction, activation of 4/8/16 frozen UOT semantic rank lists, bounded eligible lookup of 8/16/32 members per list, bounded beam construction of arity-3 tuples across distinct activated semantic edges, candidate union, and monotone QHDS selection at B=8/16/32.

An arity-3 semantic bundle is a candidate-construction object. It is not an observed co-access event, a true evidence group, or a prediction of the number of labeled evidence groups. Every online lookup records rank entries actually visited; complete eligible-by-prototype scans are prohibited.

The learned DeepSets bundle-locality r1 result is retained only as a diagnostic ablation. It may not be selected as the primary path or used to claim a quality, cost, locality, or latency gain.

## Required Comparators

ANN flat, same-capacity neural flat, dense QHDS reference, sparse semantic-bundle hypergraph, and budgeted pairwise factor graph are all mandatory. The pairwise factor graph is trained only from outer-train valid distinct-member bundles projected to unordered pairs. At inference it uses the same sparse candidate pool, protected seeds, byte accounting, Tile layout, and latency harness as the hypergraph.

## Cost And Latency

All methods use the frozen neutral source-order layout, capacity 16. Report metadata, payload, total unique, and redundant logical Tile activations. These are logical locality counters only and must not be described as PIM hardware results.

The online latency harness uses `time.perf_counter_ns()`, batch size 1, correct GPU synchronization, preloaded model/index/incidence/payload, at least 50 warm-up queries, no file I/O or logging in the timed region, raw per-query fields, and mean/p50/p95/p99 CPU and GPU summaries.

## Gate C And Follow-On Order

The core grid produces fixed-count, fixed-byte, iso-quality, iso-cost, and Pareto comparisons. Gold coverage is an explicitly nondeployable OOF oracle. Learned-utility exact solving uses a frozen utility and no gold evidence.

Gate C requires final top-5 nondegradation, no dense online scan, five-seed session-clustered paired evidence, and at least one preregistered advantage over the matched pairwise factor graph. Failure is retained and blocks any hypergraph-native superiority claim.

Only after Gate C, ablations/statistics and the dynamic feasibility decision may close. Only after all of those decisions are frozen may calibration be read exactly for CRC/LTT risk mapping without model changes. Formal test remains a single-run final operation after final freeze.
