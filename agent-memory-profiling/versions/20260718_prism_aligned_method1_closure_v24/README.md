# Method 1 v24: PRISM-Aligned Closure

This version implements the pre-reranker candidate-frontier experiment specified in `Method1_PRISM_Aligned_Closure_Codex_Guide_20260718.md`.

The closure path is:

```text
canonical memory records
-> Qwen3 embeddings
-> namespace-causal dense seeds
-> query-conditioned semantic/history hypergraph expansion
-> dense-safe top-C frontier
-> Qwen3 reranking
-> query-level quality and latency audit
```

`full_n`, `dense_c`, and `ours_c` use the same memory visibility, embeddings, reranker, prompt, maximum length, and device. Gold evidence is loaded only by the metrics layer.

## Run order

Run from the project root with the `agent-memory-profiling` Conda environment:

```bash
cd /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/00_snapshot_environment.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/01_verify_models.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/02_prepare_dataset.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/03_encode_memories.py --device cuda:0
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/05_build_history_events.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/06_build_dual_hypergraph.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/07_run_development.py --device cuda:0
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/08_freeze_config.py
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/09_run_test_once.py --device cuda:0
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/10_run_agent_memory_replay.py --device cuda:0
conda run -n agent-memory-profiling python versions/20260718_prism_aligned_method1_closure_v24/scripts/11_aggregate_closure_report.py
```

The test script refuses to read test records unless `artifacts/frozen/development_config.json` says the development Gates passed. A negative development result is still a valid closure artifact and is written under `negative_results/`.

## Contract notes

The input MedMemoryBench files are treated as immutable. History edges are made only from the train split's `flat` frozen ANN candidates and are labeled `co_candidate`; they are not native agent-read traces. Semantic prototypes are fitted with normalized soft spherical assignments and the build stops on prototype collapse. The query path does not inspect `query_type`, evidence groups, gold IDs, or answers.
