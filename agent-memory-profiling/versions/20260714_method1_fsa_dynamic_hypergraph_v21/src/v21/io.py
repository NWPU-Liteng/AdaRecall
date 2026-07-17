from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


VERSION_ROOT = Path(__file__).resolve().parents[2]
V20R1_ROOT = VERSION_ROOT / "provenance/v20_parent/v20r1"
V19_ROOT = VERSION_ROOT / "inputs/qa_basis_v1"

FORBIDDEN_TEST_MARKERS = ("query_plan_test", "formal_test", "selector_final_test", "/test/", "_test.jsonl")


def resolve(path: str | Path, *, base: Path = VERSION_ROOT) -> Path:
    item = Path(path)
    return item.resolve() if item.is_absolute() else (base / item).resolve()


def reject_test_path(path: str | Path) -> None:
    normalized = str(resolve(path)).lower()
    if any(marker in normalized for marker in FORBIDDEN_TEST_MARKERS):
        raise RuntimeError(f"v21 development cannot read formal test input: {path}")


def read_json(path: str | Path, *, allow_test: bool = False) -> Any:
    resolved = resolve(path)
    if not allow_test:
        reject_test_path(resolved)
    return json.loads(resolved.read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path, *, allow_test: bool = False) -> Iterator[dict[str, Any]]:
    resolved = resolve(path)
    if not allow_test:
        reject_test_path(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: str | Path, value: Any) -> None:
    """Create an artifact exactly once; v21 reruns use a new run or attempt ID."""
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv_new(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_attempt_directory(path: str | Path, *, allowed_existing_names: Iterable[str] = ()) -> Path:
    """Create an attempt directory without permitting artifact replacement.

    The launcher may open the current attempt log before an in-process worker
    starts. That log is the only explicitly permitted pre-existing entry.
    """
    output = resolve(path)
    allowed = set(allowed_existing_names)
    if output.exists():
        unexpected = sorted(child.name for child in output.iterdir() if child.name not in allowed)
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite existing v21 attempt directory {output}; "
                f"unexpected entries: {unexpected}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def load_config(name: str) -> dict[str, Any]:
    return read_json(VERSION_ROOT / "configs" / name)
