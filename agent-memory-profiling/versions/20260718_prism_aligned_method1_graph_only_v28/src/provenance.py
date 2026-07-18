from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_immutable_run(root: Path, run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_id):
        raise ValueError("run_id must contain only letters, digits, '_' or '-'")
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def source_manifest(paths: Iterable[Path], *, base: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(paths):
        rows.append({"path": str(path.relative_to(base)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "method1-v28-source-manifest-1", "files": rows, "manifest_sha256": canonical_sha256(rows)}
