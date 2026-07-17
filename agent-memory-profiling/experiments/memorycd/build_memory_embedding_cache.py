#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_client import TransformerEmbeddingClient
from src.memorycd.audit import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, default=Path("models/bge-base-en-v1.5"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-memories", type=int, default=None)
    args = parser.parse_args()

    memory_path = args.manifest_dir / "memory_bank.jsonl"
    if not memory_path.exists():
        raise SystemExit(f"Missing memory bank: {memory_path}")

    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ids_path = args.output_dir / "memory_embedding_ids.jsonl"
    vectors_path = args.output_dir / "memory_embeddings.npy"
    manifest_path = args.output_dir / "memory_embedding_cache_manifest.json"

    rows = []
    seen = set()
    for row in iter_jsonl(memory_path):
        memory_id = row["memory_id"]
        if memory_id in seen:
            continue
        seen.add(memory_id)
        rows.append(row)
        if args.max_memories is not None and len(rows) >= args.max_memories:
            break

    embedder = TransformerEmbeddingClient(str(args.embedding_model), dim=768, batch_size=args.batch_size)
    vectors = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        batch_vecs = embedder.embed_batch([row["memory_text"] for row in batch])
        vectors.append(batch_vecs.astype("float32"))
        if start and start % (args.batch_size * 20) == 0:
            print(json.dumps({"embedded": start, "total": len(rows)}), flush=True)
    matrix = np.vstack(vectors).astype("float32") if vectors else np.zeros((0, 768), dtype="float32")
    np.save(vectors_path, matrix)

    with ids_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            out.write(json.dumps({"memory_id": row["memory_id"], "row": idx}, ensure_ascii=False) + "\n")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    manifest = {
        "manifest_dir": str(args.manifest_dir),
        "memory_bank": str(memory_path),
        "embedding_model": str(args.embedding_model),
        "batch_size": args.batch_size,
        "memory_count": len(rows),
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 768,
        "vectors_path": str(vectors_path),
        "ids_path": str(ids_path),
        "offline_embedding_build_ms": elapsed_ms,
        "max_memories": args.max_memories,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

