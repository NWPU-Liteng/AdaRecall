# Method 1 v21 Complete Closed Loop

This document closes the static Method 1 system boundary without replacing the immutable final freeze. It combines tuned-r2 train OOF retrieval, threshold-free core KPIs, and the permitted static content-update stress. Calibration and formal test were not read.

## Correct system model

```text
OFFLINE
memory content -> embedding -> UOT fuzzy incidence (42,034 x 64)
                              -> 64 sparse prototype rank lists
memory payload ----------------> neutral capacity-16 logical Tile map
outer-train queries -----------> QESA + QHDS fitting

ONLINE, batch=1
precomputed query embedding
    |-> protected ANN top-5 --------------------------|
    `-> QESA -> active prototypes -> bounded lookup  |
                    -> arity-3 semantic bundle beam  |
                                                     v
          shared frontier (cap=96, no dense scan)
                    -> QHDS greedy order -> B=8/16/32
                    -> payload gather -> Agent handoff

STATIC CONTENT MAINTENANCE FALLBACK
new content -> frozen embedding/projection -> prototype incidence
            -> sparse overlay + append-only logical Tile
            -> atomic snapshot publication -> online lookup
delete ----> tombstone
periodic --> offline compaction -> new frozen rank-list base
```

The persistent offline structure is fuzzy prototype incidence plus rank lists. Arity-3 semantic bundles are query-conditioned candidate tuples built online; they are not a fabricated co-access hypergraph. The negative DeepSets bundle-locality result is not in the primary path, and the layout remains neutral.

## Core result at 16 activated prototypes / 32 members / beam 32 / B=32

| path | group recall | member recall | exact task success | mean e2e | p95 e2e | payload | logical Tiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| expanded frontier + QHDS | 0.687491 | 0.538553 | 0.613727 | 5.011 ms | 7.008 ms | 118689.5 B | 52.608 |
| same frontier + cosine | 0.685978 | 0.536475 | 0.613284 | 2.666 ms | 4.098 ms | 118545.5 B | 52.608 |

Compared with immutable r3, tuned-r2 changes group recall by `+0.028807` and exact task success by `+0.027601`. This is mainly the cap-32 to cap-96 frontier revision effect.

Within the exact same cap-96 frontier, QHDS versus cosine changes group recall by `+0.001513`, member recall by `+0.002077`, exact task success by `+0.000443`, mean e2e by `+2.345 ms`, payload by `+144.0 B`, and logical Tiles by `+0.000`. These are descriptive effects; no small-effect invalidation threshold is applied.

`all-groups-covered` is an exact retrieval-set success proxy, not final Agent answer accuracy. `rerank_executed=false`: the B-memory selection itself is the handoff. Logical Tiles are host-side logical counts and not PIM hardware measurements.

## Static content-update closure

The allowed maintenance stress completed growth points 1/5/10/20% with exact overlay-versus-compacted bounded-lookup agreement. At 20% growth it appended `8407` memories, used `526` append-overlay logical Tiles, built a query snapshot in `648.992 ms`, atomically published it in `5.777816 ms`, served probes at mean `76.916 ms`, and compacted the index in `2914.906 ms`. These are content-maintenance measurements over deterministic corpus slices, not causal memory-stream or access-order results.

## Operating decision without an effect threshold

- Quality-first: expanded sparse frontier plus QHDS at B=32.
- Latency-first: the same expanded sparse frontier plus cosine ordering at B=32.

This split attributes the large post-closure improvement to semantic candidate expansion while exposing the much smaller incremental selector effect. Neither operating point is labelled valid or invalid by an arbitrary minimum gain.

## Remaining boundary

The offline index, online retrieval, Agent handoff contract, and static content maintenance are closed. Final Agent answer execution is not evaluated. A true causal dynamic update experiment remains ineligible because the source has no recoverable agent read/access stream. The previous formal-test attempt remains non-reportable due to missing frozen test embeddings and is not rerun here.
