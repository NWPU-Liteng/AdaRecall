from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, V24_ROOT, VERSION_ROOT, iter_jsonl, write_json
from src.provenance import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-device", default="")
    return parser.parse_args()


def manifest_file(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    args = parse_args()
    inputs = V24_ROOT / "inputs"
    paths = {
        "memories": inputs / "memory_records.jsonl",
        "train_queries": inputs / "train_queries.jsonl",
        "train_evaluation": inputs / "train_evaluation.jsonl",
        "memory_embeddings": inputs / "embeddings" / "memory.npy",
        "query_embeddings": inputs / "embeddings" / "queries_train.npy",
        "query_index": inputs / "embeddings" / "queries_train_index.json",
    }
    counts = {
        "memories": sum(1 for _ in iter_jsonl(paths["memories"])),
        "train_queries": sum(1 for _ in iter_jsonl(paths["train_queries"])),
        "train_evaluations": sum(1 for _ in iter_jsonl(paths["train_evaluation"])),
    }
    if counts != {"memories": 42034, "train_queries": 1355, "train_evaluations": 1355}:
        raise RuntimeError(f"canonical data counts changed: {counts}")
    memories = list(iter_jsonl(paths["memories"]))
    queries = list(iter_jsonl(paths["train_queries"]))
    if len({row["namespace_id"] for row in memories}) != 20 or len({row["namespace_id"] for row in queries}) != 20:
        raise RuntimeError("expected 20 dataset-provided namespaces")
    embeddings = np.load(paths["memory_embeddings"], mmap_mode="r")
    if embeddings.shape != (42034, 1024) or not np.isfinite(np.asarray(embeddings[:128])).all():
        raise RuntimeError(f"invalid memory embedding cache: {embeddings.shape}")
    write_json(
        VERSION_ROOT / "artifacts" / "manifests" / "dataset_manifest.json",
        {
            "schema_version": "method1-v28-dataset-manifest-1",
            "dataset": "MedMemoryBench English",
            "license": "inherited from canonical v24 manifest",
            "namespace_source": "dataset_provided",
            "namespace_count": 20,
            "counts": counts,
            "forbidden_splits": ["development", "test"],
            "selection_scope": "train strict five-fold OOF development evidence",
            "files": {key: manifest_file(path) for key, path in paths.items()},
        },
    )
    model_rows = {}
    for name in ("Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B"):
        root = PROJECT_ROOT / "models" / "models" / name
        required = [root / "config.json", root / "tokenizer.json", root / "model.safetensors"]
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"incomplete local model checkpoint: {root}")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        model_rows[name] = {
            "path": str(root),
            "config": manifest_file(root / "config.json"),
            "tokenizer": manifest_file(root / "tokenizer.json"),
            "weights": manifest_file(root / "model.safetensors"),
            "model_type": config.get("model_type"),
            "local_files_only": True,
            "attention_implementation": "eager",
        }
    if args.smoke_device:
        import torch
        from src.qwen_embedding import Qwen3Embedding
        from src.qwen_reranker import Qwen3Reranker

        if not torch.cuda.is_available():
            raise RuntimeError("formal Qwen smoke requires CUDA")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(args.smoke_device)
        started = time.perf_counter()
        embedder = Qwen3Embedding(PROJECT_ROOT / "models" / "models" / "Qwen3-Embedding-0.6B", device=args.smoke_device, dtype="bfloat16", max_length=512, batch_size=2)
        embedding_cold_start_ms = (time.perf_counter() - started) * 1000.0
        smoke = embedder.smoke()
        texts = ["Valsartan treats hypertension.", "A train ticket was booked."]
        started = time.perf_counter()
        batch = embedder.encode_documents(texts)
        embedding_warm_ms = (time.perf_counter() - started) * 1000.0
        singles = np.vstack([embedder.encode_documents([text])[0] for text in texts])
        embedding_consistency = float(np.max(np.abs(batch - singles)))
        model_rows["Qwen3-Embedding-0.6B"]["smoke"] = {
            **smoke,
            "batch_single_max_abs": embedding_consistency,
            "batch_single_passed": embedding_consistency <= 2e-3,
            "cold_start_ms": embedding_cold_start_ms,
            "warm_batch_ms": embedding_warm_ms,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(args.smoke_device)),
        }
        if not smoke["passed"] or embedding_consistency > 2e-3:
            raise RuntimeError("Qwen embedding smoke failed")
        del embedder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.smoke_device)

        started = time.perf_counter()
        reranker = Qwen3Reranker(PROJECT_ROOT / "models" / "models" / "Qwen3-Reranker-0.6B", device=args.smoke_device, dtype="bfloat16", max_length=512, batch_size=2)
        reranker_cold_start_ms = (time.perf_counter() - started) * 1000.0
        reranker_smoke = reranker.smoke()
        query = "Which medication treats hypertension?"
        started = time.perf_counter()
        batch_scores = np.asarray(reranker.score_pairs(query, texts).scores)
        reranker_warm_ms = (time.perf_counter() - started) * 1000.0
        single_scores = np.asarray([reranker.score_pairs(query, [text]).scores[0] for text in texts])
        reranker_consistency = float(np.max(np.abs(batch_scores - single_scores)))
        model_rows["Qwen3-Reranker-0.6B"]["smoke"] = {
            **reranker_smoke,
            "batch_single_max_abs": reranker_consistency,
            "batch_single_passed": reranker_consistency <= 2e-3,
            "cold_start_ms": reranker_cold_start_ms,
            "warm_batch_ms": reranker_warm_ms,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(args.smoke_device)),
        }
        if not reranker_smoke["passed"] or reranker_consistency > 2e-3:
            raise RuntimeError("Qwen reranker smoke failed")
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "models_manifest.json", {"schema_version": "method1-v28-model-manifest-1", "models": model_rows})
    print("GATE_A_DATA_MODEL_FILES_VERIFIED")


if __name__ == "__main__":
    main()
