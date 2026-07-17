#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Iterable


def ensure_output_layout(output_dir: Path) -> Dict[str, Path]:
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "logs_dir": logs_dir,
        "raw_profile": output_dir / "raw_profile.jsonl",
        "phase_trace": output_dir / "phase_trace.jsonl",
        "summary_csv": output_dir / "summary.csv",
        "summary_md": output_dir / "summary.md",
    }


def append_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
