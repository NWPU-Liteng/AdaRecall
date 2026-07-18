from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.embedding import Qwen3Embedding  # noqa: E402
from v24.io import iter_jsonl, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    canonical = VERSION_ROOT / "artifacts" / "fresh" / "canonical"
    receipt = json.loads((VERSION_ROOT / "artifacts" / "fresh" / "fresh_read_receipt.json").read_text())
    if receipt.get("status") != "COMPLETED_GOLD_READ_ONCE":
        raise RuntimeError("fresh source has not completed its single read")
    memories = list(iter_jsonl(canonical / "memory_records.jsonl"))
    queries = list(iter_jsonl(canonical / "queries.jsonl"))
    output = VERSION_ROOT / "artifacts" / "fresh" / "embeddings"
    output.mkdir(parents=True, exist_ok=True)
    model = Qwen3Embedding(
        PROJECT_ROOT / "models" / "models" / "Qwen3-Embedding-0.6B",
        device=args.device,
        dtype="bfloat16",
        max_length=512,
        batch_size=16,
    )
    if not model.device.startswith("cuda"):
        raise RuntimeError("fresh embedding requires CUDA")
    smoke = model.smoke()
    if not smoke["passed"]:
        raise RuntimeError(f"embedding smoke failed: {smoke}")
    memory_path = output / "memory.npy"
    query_path = output / "queries.npy"
    if not memory_path.exists():
        np.save(memory_path, model.encode_documents([row["content"] for row in memories]))
    if not query_path.exists():
        np.save(query_path, model.encode_queries([row["query"] for row in queries]))
    write_json(
        output / "manifest.json",
        {
            "memory_count": len(memories),
            "query_count": len(queries),
            "memory_ids": [row["memory_id"] for row in memories],
            "query_ids": [row["query_id"] for row in queries],
            "runtime": model.runtime.__dict__,
            "attention_implementation": model.attention_implementation,
            "smoke": smoke,
        },
    )
    print(json.dumps({"memory_count": len(memories), "query_count": len(queries), "runtime": model.runtime.__dict__}, sort_keys=True))


if __name__ == "__main__":
    main()
