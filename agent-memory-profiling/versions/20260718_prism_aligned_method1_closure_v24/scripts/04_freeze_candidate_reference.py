from __future__ import annotations

import argparse
import json

import numpy as np

from common import VERSION_ROOT, add_common_args, config_hash, load_config, save_resolved_config
from v24.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    embedding_manifest_path = VERSION_ROOT / "artifacts" / "manifests" / "embedding_manifest.json"
    if not embedding_manifest_path.exists():
        raise FileNotFoundError("run 03_encode_memories.py before freezing candidate references")
    manifest = read_json(embedding_manifest_path)
    if int(manifest["records"]["memory"]["dimension"]) != 1024:
        raise RuntimeError("Qwen3 memory embedding dimension is not 1024")
    reference = {
        "schema_version": "method1-v24-candidate-reference-1",
        "config_hash": config_hash(config),
        "embedding_backend": config["model"]["embedding_repo"],
        "reranker_backend": config["model"]["reranker_repo"],
        "sanity": {
            "full_n": config["budget"]["sanity_full_n"],
            "frontier_c": config["budget"]["sanity_frontier_c"],
            "final_k": config["budget"]["sanity_final_k"],
        },
        "main": {
            "full_n": config["budget"]["full_n"],
            "seed_s": config["budget"]["seed_s"],
            "frontier_c": config["budget"]["frontier_c"],
            "final_k": config["budget"]["final_k"],
        },
        "fairness": [
            "same visible namespace and causal cutoff",
            "same memory/query embeddings",
            "same Qwen3 reranker prompt, max length, dtype, and batch size",
            "Ours edge activation and diffusion included in semantic_selection_ms",
        ],
        "test_read_status": "not_read",
    }
    write_json(VERSION_ROOT / "artifacts" / "frozen" / "candidate_reference.json", reference)
    print(json.dumps({"status": "completed", "config_hash": reference["config_hash"], "main": reference["main"]}, sort_keys=True))


if __name__ == "__main__":
    main()
