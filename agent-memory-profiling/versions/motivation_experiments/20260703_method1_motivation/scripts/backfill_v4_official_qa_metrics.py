#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent.parent
MANIFEST_ROOT = PROJECT_ROOT / "versions" / "20260621_real_trace_adaptive_phase_v4_formal" / "manifests"

import sys

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "motivation_memory_selection"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.append(str(EXPERIMENTS_ROOT))

from qa_eval_utils import score_prediction


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_reference_answers() -> dict[str, dict[str, Any]]:
    answers: dict[str, dict[str, Any]] = {}
    for dataset in ["locomo", "longmemeval_s"]:
        for row in _read_jsonl(MANIFEST_ROOT / dataset / "evaluation_labels.jsonl"):
            answers[row["event_id"]] = row
    return answers


def _backfill_file(path: Path, labels: dict[str, dict[str, Any]]) -> tuple[int, int]:
    rows = _read_jsonl(path)
    updated = 0
    for row in rows:
        if row.get("selection_only") is True:
            continue
        label = labels.get(row["event_id"])
        if label is None:
            continue
        qa_scores = score_prediction(
            row.get("answer_text", ""),
            label.get("reference_answer"),
            query_text=row.get("query_text"),
        )
        changed = False
        for key, value in qa_scores.items():
            if row.get(key) != value:
                row[key] = value
                changed = True
        if changed:
            updated += 1
    if updated:
        _write_jsonl(path, rows)
    return len(rows), updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill richer QA metrics into existing V4 official-QA raw profiles.")
    parser.add_argument("--root", default=str(VERSION_ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    labels = _load_reference_answers()

    total_rows = 0
    total_updated = 0
    for raw_path in sorted((root / "runs").glob("v4_*_budget*/raw_profile.jsonl")):
        rows, updated = _backfill_file(raw_path, labels)
        total_rows += rows
        total_updated += updated
        print(f"backfilled={updated} rows={rows} path={raw_path}")
    print(f"total_rows={total_rows}")
    print(f"total_updated={total_updated}")


if __name__ == "__main__":
    main()
