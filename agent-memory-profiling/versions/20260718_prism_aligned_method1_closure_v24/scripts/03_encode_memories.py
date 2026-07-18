from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT, VERSION_ROOT, add_common_args, load_config, resolve_project_path
from v24.embedding import Qwen3Embedding
from v24.io import iter_jsonl, write_json


def _encode_memory_stream(encoder: Qwen3Embedding, source: Path, output: Path, ids_path: Path, batch_size: int) -> dict:
    count = sum(1 for _ in iter_jsonl(source))
    dimension = int(encoder.runtime.dimension)
    if dimension <= 0:
        probe = encoder.encode_documents(["dimension probe"])
        dimension = int(probe.shape[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(count, dimension))
    ids = []
    batch_texts = []
    batch_ids = []
    row_index = 0
    for row in iter_jsonl(source):
        batch_texts.append(str(row.get("text", row.get("content", ""))))
        batch_ids.append(str(row["memory_id"]))
        if len(batch_texts) < int(batch_size) and row_index + len(batch_texts) < count:
            continue
        values = encoder.encode_documents(batch_texts)
        matrix[row_index:row_index + len(values)] = values
        ids.extend(batch_ids)
        row_index += len(values)
        batch_texts, batch_ids = [], []
    matrix.flush()
    write_json(ids_path, ids)
    return {"rows": count, "dimension": dimension, "dtype": "float32", "path": str(output), "ids_path": str(ids_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-memories", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    model_dir = PROJECT_ROOT / config["model"]["embedding_model_dir"]
    encoder = Qwen3Embedding(
        model_dir, device=args.device, dtype=config["model"]["dtype"], max_length=config["model"]["max_length"],
        instruction=config["model"]["instruction"], batch_size=args.batch_size,
    )
    output_root = VERSION_ROOT / "inputs" / "embeddings"
    output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    if not args.skip_memories:
        records["memory"] = _encode_memory_stream(
            encoder, VERSION_ROOT / "inputs" / "memory_records.jsonl", output_root / "memory.npy", output_root / "memory_ids.json", args.batch_size,
        )
    for split in ("train", "development", "test"):
        rows = list(iter_jsonl(VERSION_ROOT / "inputs" / f"{split}_queries.jsonl"))
        values = encoder.encode_queries([str(row["query"]) for row in rows])
        np.save(output_root / f"queries_{split}.npy", values.astype(np.float32), allow_pickle=False)
        index = {"dimension": int(values.shape[1]), "embedding_backend": "Qwen/Qwen3-Embedding-0.6B", "records": [{"query_id": str(row["query_id"]), "query": str(row["query"])} for row in rows]}
        write_json(output_root / f"queries_{split}_index.json", index)
        records[f"queries_{split}"] = {"rows": len(rows), "dimension": int(values.shape[1]), "path": str(output_root / f"queries_{split}.npy")}
    manifest = {
        "model_repo": config["model"]["embedding_repo"], "model_dir": str(model_dir), "runtime": encoder.runtime.__dict__,
        "max_length": config["model"]["max_length"], "instruction": config["model"]["instruction"], "records": records,
    }
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "embedding_manifest.json", manifest)
    print(json.dumps({"status": "completed", "runtime": encoder.runtime.__dict__, "records": records}, sort_keys=True))


if __name__ == "__main__":
    main()
