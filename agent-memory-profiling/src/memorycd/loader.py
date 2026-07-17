from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import DOMAINS


def open_text_auto(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_user_records(users_path: Path) -> Iterable[dict[str, Any]]:
    with open_text_auto(users_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metadata_sample(meta_dir: Path, limit_per_domain: int = 3) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        path = meta_dir / f"{domain}.jsonl.gz"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with open_text_auto(path) as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
                    if len(rows) >= limit_per_domain:
                        break
        samples[domain] = rows
    return samples


def split_last_n(interactions: list[dict[str, Any]], num_test: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(interactions, key=lambda row: int(row.get("timestamp") or 0))
    if len(ordered) <= num_test:
        if len(ordered) <= 1:
            return [], ordered
        return ordered[:-1], ordered[-1:]
    return ordered[:-num_test], ordered[-num_test:]


def safe_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, list):
        text = " ".join(str(item) for item in value if item is not None)
    else:
        text = str(value)
    text = " ".join(text.replace("<br />", " ").split())
    if limit is not None and len(text) > limit:
        return text[:limit] + "..."
    return text

