from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import PROJECT_ROOT, VERSION_ROOT, add_common_args, load_config
from v24.embedding import Qwen3Embedding
from v24.io import sha256_file, write_json
from v24.reranker import Qwen3Reranker


def model_manifest(path: Path, repo: str) -> dict:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".cache" not in item.parts:
            files.append({"path": str(item.relative_to(path)), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    return {"repo_id": repo, "local_dir": str(path), "present": path.is_dir(), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--skip-load", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    embedding_dir = PROJECT_ROOT / config["model"]["embedding_model_dir"]
    reranker_dir = PROJECT_ROOT / config["model"]["reranker_model_dir"]
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding": model_manifest(embedding_dir, config["model"]["embedding_repo"]),
        "reranker": model_manifest(reranker_dir, config["model"]["reranker_repo"]),
        "dtype": config["model"]["dtype"], "attention": config["model"]["attention"],
        "max_length": config["model"]["max_length"], "local_files_only": True,
    }
    if not args.skip_load:
        if not embedding_dir.is_dir() or not reranker_dir.is_dir():
            raise FileNotFoundError("official Qwen3 model directories are incomplete")
        embedding = Qwen3Embedding(
            embedding_dir, device=args.device, dtype=config["model"]["dtype"], max_length=config["model"]["max_length"],
            instruction=config["model"]["instruction"], batch_size=2,
        )
        reranker = Qwen3Reranker(
            reranker_dir, device=args.device, dtype=config["model"]["dtype"], max_length=config["model"]["max_length"],
            batch_size=2, instruction=config["model"]["instruction"],
        )
        manifest["embedding_smoke"] = embedding.smoke()
        manifest["reranker_smoke"] = reranker.smoke()
        manifest["runtime_device"] = {"embedding": embedding.device, "reranker": reranker.device}
        manifest["passed"] = bool(manifest["embedding_smoke"]["passed"] and manifest["reranker_smoke"]["passed"])
    else:
        manifest["passed"] = False
        manifest["load_skipped"] = True
    write_json(VERSION_ROOT / "artifacts" / "models" / "models_manifest.json", manifest)
    print({"status": "completed", "passed": manifest["passed"], "embedding": str(embedding_dir), "reranker": str(reranker_dir)})


if __name__ == "__main__":
    main()
