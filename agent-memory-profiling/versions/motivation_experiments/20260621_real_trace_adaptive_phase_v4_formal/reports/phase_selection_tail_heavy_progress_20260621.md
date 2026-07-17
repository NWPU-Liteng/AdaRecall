# Phase Selection Tail-Heavy Progress

## Scope

This note records the tail-heavy `official_qa + selection_only + real local Qwen` follow-up runs completed on June 21, 2026.

## Why the previous slice was insufficient

- The earlier LoCoMo `official_qa` comparison used the first 16 rows that satisfied `min_visible_memory_count >= 128`.
- Those rows all had `available_memory_count = 419`, which is far below the LoCoMo `official_qa` tail (`max = 689`).
- As a result, the selection path was real but under-stressed.

## Code change

- Added `event_sort_by` support in `experiments/motivation_memory_selection/profile_real_agent_v4.py`.
- Supported values:
  - `visible_memory_count_desc`
  - `visible_memory_count_asc`
- This lets configs explicitly select the heaviest real events instead of depending on manifest order.

## LoCoMo tail-heavy setup

- dataset: `locomo`
- event type: `official_qa`
- selection only: `true`
- generation backend: real local `Qwen`
- retriever / reranker:
  - `retrieval_top_r = 256`
  - `pruner_top_b = 64`
  - `final_top_k = 16`
- event filter:
  - `min_visible_memory_count = 689`
  - `event_sort_by = visible_memory_count_desc`
  - `max_events = 32`

## LoCoMo tail-heavy results

| mode | runs | avg retrieval count | avg selection ms | avg controller ms | avg total ms |
|---|---:|---:|---:|---:|---:|
| shared_once | 32 | 1.0 | 522.84 | 0.00 | 541.38 |
| fixed_three | 32 | 3.0 | 1632.47 | 0.00 | 1650.79 |
| adaptive_phase | 32 | 2.0 | 958.39 | 1696.46 | 2678.29 |

## Interpretation

- The tail-heavy slice clearly increases real selection pressure:
  - `shared_once`: `297.25 ms -> 522.84 ms`
  - `fixed_three`: `913.54 ms -> 1632.47 ms`
  - `adaptive_phase`: `565.82 ms -> 958.39 ms`
- The selection-path ranking is now stable and interpretable:
  - `shared_once < adaptive_phase < fixed_three`
- The adaptive controller is still the main reason end-to-end latency stays high:
  - average controller cost remains about `1.7 s`
  - average retrieval count remains `2.0`
- Therefore the current setup supports a selection-path bottleneck analysis, but not yet a strong end-to-end latency win claim for adaptive mode.

## Next step

- Replicate the same tail-heavy protocol on `LongMemEval-S official_qa`.
- Keep the same real Qwen controller and the same retriever / reranker stack.
- Do not introduce heuristic or synthetic controller replacements.

## LongMemEval-S tail-heavy setup

- dataset: `longmemeval_s`
- event type: `official_qa`
- selection only: `true`
- generation backend: real local `Qwen`
- retriever / reranker:
  - `retrieval_top_r = 256`
  - `pruner_top_b = 64`
  - `final_top_k = 16`
- event filter:
  - `min_visible_memory_count = 540`
  - `event_sort_by = visible_memory_count_desc`
  - `max_events = 32`

## LongMemEval-S tail-heavy results

| mode | runs | avg retrieval count | avg selection ms | avg controller ms | avg total ms |
|---|---:|---:|---:|---:|---:|
| shared_once | 32 | 1.0 | 512.46 | 0.00 | 512.98 |
| fixed_three | 32 | 3.0 | 1589.25 | 0.00 | 1590.56 |
| adaptive_phase | 32 | 2.0 | 998.40 | 1546.24 | 2550.51 |

## Cross-dataset interpretation

- The selection-path ordering is consistent across both real datasets:
  - `shared_once < adaptive_phase < fixed_three`
- The adaptive controller keeps the expected online behavior:
  - planning phase retrieves
  - reasoning phase retrieves
  - output phase skips
  - total retrieval count remains `2.0`
- The main obstacle to an end-to-end latency win is still controller cost, not fake selection pressure:
  - LoCoMo adaptive controller: `1696.46 ms`
  - LongMemEval-S adaptive controller: `1546.24 ms`
- Therefore the current evidence is now strong enough to support:
  - real phase-dependent selection cost exists on official QA workloads
  - heavier true memory slices amplify the selection bottleneck substantially
  - adaptive retrieval reduces selection cost relative to fixed-three without collapsing to shared-once
- The current evidence is still not enough to claim:
  - adaptive mode has a full end-to-end latency advantage under the present real-Qwen controller design

## Recommended next step

- Treat `selection_only + tail-heavy official_qa` as the main bottleneck-analysis setting.
- Use these runs as the basis for the latency-breakdown figure and system-cost table.
- Then move to the next V4 stage:
  - phase-locality structure analysis on real traces
  - online multiview hypergraph pruning

## Follow-up adjustment

- `selection_only=true` should not execute the `output` phase.
- Rationale:
  - in selection-only mode there is no final answer generation
  - keeping `output` would add an extra controller decision without a downstream consumer
  - this inflates the denominator of the bottleneck experiment
- The profiler has now been updated so selection-only runs execute only:
  - `planning`
  - `reasoning`
