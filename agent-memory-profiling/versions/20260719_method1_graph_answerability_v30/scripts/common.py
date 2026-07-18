from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

VERSION_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = VERSION_ROOT.parent
PROJECT_ROOT = VERSIONS_ROOT.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
V28_ROOT = VERSIONS_ROOT / "20260718_prism_aligned_method1_graph_only_v28"
V24_ROOT = VERSIONS_ROOT / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V24_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value)!r}")


def write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, default=_json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=_json_default) + "\n")


def create_immutable_run(run_id: str) -> Path:
    if not run_id or any(value in run_id for value in ("/", "\\", "..")):
        raise ValueError("invalid run ID")
    path = VERSION_ROOT / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path
