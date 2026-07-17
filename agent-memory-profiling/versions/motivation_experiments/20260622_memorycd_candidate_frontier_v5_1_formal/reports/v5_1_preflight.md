# V5.1 Preflight

日期：2026-06-22

## Environment

- Conda env: `agent-memory-profiling`
- Python: `3.10.20`
- Torch: `2.1.0+cu121`
- CUDA availability under authorized execution: `true`
- CUDA visible devices under authorized execution: `8`
- GPU: `NVIDIA GeForce RTX 3090`
- Transformers: `4.51.3`
- NumPy: `1.26.4`
- FAISS: `1.9.0`

## Data

MemoryCD data has been moved under:

```text
data/public_datasets/memorycd/
├── meta/
│   ├── Beauty_and_Personal_Care.jsonl.gz
│   ├── Books.jsonl.gz
│   ├── Electronics.jsonl.gz
│   └── Home_and_Kitchen.jsonl.gz
└── users/
    └── cross_domain_users_sampled.jsonl.gz
```

The local dataset uses a path mapping rather than copying or rewriting files:

- `users/cross_domain_users_sampled.jsonl.gz` is the unified user interaction file.
- `meta/<Domain>.jsonl.gz` files are the item metadata files.

Inventory and SHA256 hashes are saved in `memorycd_dataset_inventory.json`.

## Model Paths

- Dense retriever: `models/bge-base-en-v1.5`
- Cross-encoder: `models/cross-encoder-ms-marco-MiniLM-L6-v2`
- Local Qwen: `../dir`

## Isolation From Previous Runs

V5.1 uses new code under:

- `src/memorycd/`
- `experiments/memorycd/`
- `versions/20260622_memorycd_candidate_frontier_v5_1_formal/`

It does not reuse the old V4 phase-aware controller path and does not write into V4/V5 result directories.

## Current Limitation

The smoke selection runner currently computes memory embeddings online per event. This is acceptable for validating correctness and trace schema, but not acceptable as formal online latency. Formal runs must add an offline embedding cache first.

