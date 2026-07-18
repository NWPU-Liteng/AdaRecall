# Gate A: Model Acquisition Resolved

This file supersedes the earlier blocked note. The official local snapshots are now present and Gate A passed on 2026-07-18.

- Embedding: `models/models/Qwen3-Embedding-0.6B/`
- Reranker: `models/models/Qwen3-Reranker-0.6B/`
- Both checkpoints loaded with `local_files_only=true` on `cuda:0`.
- Embedding smoke passed with dimension `1024`, finite L2-normalized output, and positive-vs-negative similarity ordering.
- Reranker smoke passed with valid distinct `yes/no` token IDs and positive-vs-negative score ordering.
- The configured SDPA path was rejected by `transformers==4.51.3` with `torch==2.1.0` because that combination requires `torch>=2.1.1`; the wrapper recorded an explicit `eager` fallback rather than upgrading PyTorch/CUDA.

Full evidence is in `artifacts/models/models_manifest.json`; the formal development, test-once, and replay runs completed under the same fallback.
