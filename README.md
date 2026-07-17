# AdaRecall

AdaRecall is the Method 1 research implementation for budget-constrained
access-basis hypergraph compilation in agent-memory systems. It optimizes
candidate-frontier construction between dense retrieval and final
cross-encoder reranking, with explicit coverage, overfetch, tile-capacity, and
replication constraints.

## Repository layout

- `agent-memory-profiling/src/`: shared profiling and retrieval components.
- `agent-memory-profiling/scripts/`: baseline experiment entry points.
- `agent-memory-profiling/versions/20260714_method1_fsa_dynamic_hypergraph_v21/`:
  canonical FSA-QEHG implementation, tests, configurations, and reports.
- `agent-memory-profiling/versions/INDEX.md`: experiment-version and
  consolidation index.
- `AdaRecall_Method1_Codex_Execution_Guide.md`: Method 1 design and execution
  guide.
- `guide/`: supporting experiment and handoff documentation.

Start with the canonical package's
[`CONSOLIDATION_20260717.md`](agent-memory-profiling/versions/20260714_method1_fsa_dynamic_hypergraph_v21/CONSOLIDATION_20260717.md)
and review packet under its `reports/` directory.

## Environment

Install the Python dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

The validated formal experiment environment is named
`agent-memory-profiling`. See the execution guide and package configs before
starting GPU-backed runs.

## Data and generated artifacts

Downloaded datasets, local model weights, embeddings, checkpoints, runtime
manifests, and large experiment outputs are intentionally excluded from Git.
They remain local or in the project's external artifact storage. The tracked
repository contains the code, configs, tests, reports, and compact summaries
needed to inspect and reproduce the pipeline once those inputs are provisioned.
