#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_client import TransformerEmbeddingClient
from src.memorycd.audit import iter_jsonl
from src.memorycd.faiss_store import brute_force_ip, search_flat_ip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--r", type=int, default=128)
    args = parser.parse_args()
    os.environ["LOCAL_QWEN_CUDA_DEVICE"] = str(args.gpu)

    manifest_dir = args.version_dir / "manifests" / f"memorycd_{args.protocol}"
    memory_by_id = {row["memory_id"]: row for row in iter_jsonl(manifest_dir / "memory_bank.jsonl")}
    event = next(row for row in iter_jsonl(manifest_dir / "runtime_manifest.jsonl") if row["event_id"] == args.event_id)
    row_by_id = {row["memory_id"]: int(row["row"]) for row in iter_jsonl(args.embedding_cache_dir / "memory_embedding_ids.jsonl")}
    matrix = np.load(args.embedding_cache_dir / "memory_embeddings.npy", mmap_mode="r")
    ids = [mid for mid in event["runtime_memory_ids"] if mid in row_by_id and mid in memory_by_id]
    rows = [row_by_id[mid] for mid in ids]
    vectors = np.asarray(matrix[rows], dtype="float32")
    embedder = TransformerEmbeddingClient("models/bge-base-en-v1.5", dim=768, batch_size=16)
    query = embedder.embed(event["query_text"])
    faiss_scores, faiss_idx = search_flat_ip(vectors, query, args.r)
    brute_idx = brute_force_ip(vectors, query, args.r)
    faiss_ids = [ids[int(i)] for i in faiss_idx]
    brute_ids = [ids[int(i)] for i in brute_idx]
    scores = vectors @ query.astype("float32")
    first_diff = next((i for i, (a, b) in enumerate(zip(faiss_ids, brute_ids)) if a != b), None)
    report = {
        "event_id": args.event_id,
        "N": len(ids),
        "R": args.r,
        "first_diff": first_diff,
        "faiss_top10": [{"id": faiss_ids[i], "score": float(scores[int(faiss_idx[i])])} for i in range(min(10, len(faiss_ids)))],
        "brute_top10": [{"id": brute_ids[i], "score": float(scores[int(brute_idx[i])])} for i in range(min(10, len(brute_ids)))],
    }
    if first_diff is not None:
        fi = int(faiss_idx[first_diff])
        bi = int(brute_idx[first_diff])
        report["diff_detail"] = {
            "rank": first_diff,
            "faiss_id": ids[fi],
            "faiss_score": float(scores[fi]),
            "brute_id": ids[bi],
            "brute_score": float(scores[bi]),
            "abs_score_delta": float(abs(scores[fi] - scores[bi])),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
