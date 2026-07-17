#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.embedding_client import LocalQwenEmbeddingClient, OllamaEmbeddingClient
from src.vector_store import get_vector_store


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--prefer-qdrant", type=int, default=1)
    parser.add_argument("--retrieval-backend", choices=["auto", "numpy", "ann_numpy", "qdrant"], default="auto")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--embedding-backend", choices=["auto", "local_qwen", "ollama_hash"], default="auto")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/build_index.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_path = Path(args.input)
    records = load_jsonl(input_path)
    size = len(records)

    if args.embedding_backend == "local_qwen":
        embed_client = LocalQwenEmbeddingClient(model_dir=os.getenv("LOCAL_QWEN_DIR", "../dir"), dim=args.dim)
    else:
        if args.embedding_backend == "auto" and os.getenv("LOCAL_QWEN_DIR"):
            try:
                embed_client = LocalQwenEmbeddingClient(model_dir=os.getenv("LOCAL_QWEN_DIR"), dim=args.dim)
            except Exception:
                embed_client = OllamaEmbeddingClient(dim=args.dim)
        else:
            embed_client = OllamaEmbeddingClient(dim=args.dim)
    vectors = embed_client.embed_batch(row["content"] for row in records)

    store = get_vector_store(
        bool(args.prefer_qdrant),
        size=size,
        dim=args.dim,
        base_dir="data",
        retrieval_backend=args.retrieval_backend,
    )
    backend_mode = getattr(store, "mode", "numpy_fallback")
    if backend_mode == "qdrant":
        try:
            store.build(vectors, records)
        except Exception:
            fallback_backend = "ann_numpy" if args.retrieval_backend == "auto" else "numpy"
            store = get_vector_store(
                False,
                size=size,
                dim=args.dim,
                base_dir="data",
                retrieval_backend=fallback_backend,
            )
            backend_mode = store.mode
            store.build(vectors, records)
    else:
        store.build(vectors, records)

    status = {
        "backend_mode": backend_mode,
        "embedding_mode": embed_client.mode,
        "num_memories": size,
        "dim": args.dim,
        "input_file": str(input_path),
        "retrieval_backend": args.retrieval_backend,
    }
    with open("results/index_status.json", "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, ensure_ascii=False)
    logging.info("index built: %s", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
