from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = VERSION_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from v24.io import read_json, sha256_file, write_json  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(VERSION_ROOT / "configs" / "closure_main.json"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", default="auto")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = read_json(config_path)
    config["config_path"] = str(config_path.resolve())
    config["version_root"] = str(VERSION_ROOT)
    config["project_root"] = str(PROJECT_ROOT)
    try:
        config["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        config["git_commit"] = "unknown"
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def config_hash(config: Mapping[str, Any]) -> str:
    value = {key: config[key] for key in sorted(config) if key not in {"config_path", "version_root", "project_root", "git_commit"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    value = dict(config)
    value["config_hash"] = config_hash(config)
    write_json(path, value)


def data_root() -> Path:
    return VERSION_ROOT / "inputs"


def artifact_root(run_id: str) -> Path:
    path = VERSION_ROOT / "runs" / run_id / "index"
    path.mkdir(parents=True, exist_ok=True)
    return path
