# Method 1 PRISM-Aligned Closure Report

## Status: `METHOD1_CLOSURE_FAILED_OR_INCOMPLETE`

This report preserves negative or incomplete Gates. It does not promote a method based on a development-only or dense-reference result.

## Gate State

- Gate A environment/model: `True`
- Gate B data/leakage: `True`
- development quality/cost Gate: `True`
- formal test: `COMPLETED_ONCE`
- formal test quality: `False`
- formal test net cost: `True`
- offline Agent Memory replay: `True`

## Frozen Configuration

- N/S/C/K: `128/16/32/5`
- semantic prototypes/top memberships: `128/4`
- diffusion alpha/steps/beta: `0.8/2/0.7`
- candidate reduction: `0.750`
- model attention configured/actual: `sdpa/eager/eager`
- model loading: `local_files_only=true`, embedding and reranker dimensions/checks in `artifacts/models/models_manifest.json`
- dataset: existing MedMemoryBench English inputs; no new dataset was downloaded for this closure.

## Formal Test Paired Summary

- paired queries: `290`; test status: `COMPLETED_ONCE`; read: `read_once`
- Ours-C minus Dense-C nDCG@10: `-0.0021756198399015777`; 95% bootstrap CI `[-0.006526859519704733, 0.0]`; frozen epsilon `-0.005`
- Ours-C minus Full-N nDCG@10: `-0.002970183159126849`; 95% bootstrap CI `[-0.007425457897817122, 0.0]`; frozen epsilon `-0.01`
- Full-N minus Ours-C semantic-selection net gain (ms): `2329.753160137876`; 95% bootstrap CI `[2324.8565970513378, 2335.6071452817137]`
- Strict Gate E quality checks: `{'dense_mean_non_negative': False, 'dense_ci_lower_within_epsilon': False, 'full_ci_lower_within_epsilon': True}`.
- The formal test quality gate is not promoted: Ours-C has a negative mean nDCG delta versus Dense-C and its paired bootstrap lower bound crosses the frozen Dense-C epsilon. No test rerun or post-test tuning was performed.

## Main Table

| Method | N | C | K | Precision@K | nDCG@10 | Ref overlap | Method ms | Rerank ms | Total ms | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_n | 128 | 128 | 5 | 0.0014 | 0.0030 | 1.0000 | 0.0000 | 3121.6987 | 3124.8293 | 3340361728.0000 |
| dense_c | 128 | 32 | 5 | 0.0007 | 0.0022 | 0.2814 | 0.0000 | 785.0082 | 787.7157 | 3340361728.0000 |
| ours_c | 128 | 32 | 5 | 0.0000 | 0.0000 | 0.3055 | 6.7980 | 785.7328 | 795.0762 | 3340361728.0000 |

## Agent Memory Replay

- tasks/events: `128`; task success available: `False`; source semantics: `co_candidate`
- `full_n`: final reference agreement `0.012939453125`, frontier agreement `0.206298828125`, pairs `16384`, semantic-selection sum ms `400034.66456593014`
- `dense_c`: final reference agreement `0.012451171875`, frontier agreement `0.056884765625`, pairs `4096`, semantic-selection sum ms `101098.55424496345`
- `ours_c`: final reference agreement `0.01318359375`, frontier agreement `0.062255859375`, pairs `4096`, semantic-selection sum ms `102101.0105824098`

## Artifacts

- development results: `runs/` matching `v24_development_*/query_results.jsonl`
- test result: `/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260718_prism_aligned_method1_closure_v24/runs/v24_test_once_20260718/query_results.jsonl`
- canonical data and split manifests: `artifacts/manifests/`
- semantic/history diagnostics: `artifacts/hypergraph/`
- machine-readable state: `aggregate/closure_state.json`
- figures: `figures/`

## Limitations and Negative Cases

- The history view is derived from frozen train ANN candidate lists and is labeled `co_candidate`; it is not a native agent-read trace.
- The selector does not inspect query type or gold evidence. Those fields are loaded only by the evaluation layer.
- The configured SDPA attention path was unavailable with torch 2.1.0 and transformers 4.51.3; both official Qwen3 wrappers explicitly recorded an eager-attention fallback. No PyTorch/CUDA upgrade was performed.
- Absolute quality values are sparse on this workload; this version does not claim a passed quality closure. The reproducible positive signal is development/replay semantic-selection speed, while the formal test quality gate remains failed.
- If the environment cannot load the official models or expose a usable GPU, the run remains an implementation/smoke artifact and is not a formal result.
- No test configuration is changed after the first formal test read.
