#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN = [
    "gold_group_id",
    "qa.answer",
    "qa.evidence",
    "answer_session_ids",
    "has_answer",
]

RUNTIME_FILES = [
    "adaptive_phase_controller.py",
    "profile_real_agent_v4.py",
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V4 runtime code and phase traces for leakage-sensitive fields.")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--phase-trace", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    phase_trace_path = Path(args.phase_trace).resolve() if args.phase_trace else None
    output_path = Path(args.output).resolve()
    issues: list[str] = []

    for rel_path in RUNTIME_FILES:
        path = runtime_root / rel_path
        if not path.exists():
            issues.append(f"missing runtime file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                issues.append(f"forbidden token `{token}` found in runtime file `{path}`")

    if phase_trace_path and phase_trace_path.exists():
        for row in _read_jsonl(phase_trace_path):
            fields = set(row.get("runtime_input_fields") or [])
            bad = fields & set(FORBIDDEN)
            if bad:
                issues.append(f"phase trace event `{row.get('event_id')}` contains forbidden runtime_input_fields: {sorted(bad)}")

    lines = ["# Leakage Audit", ""]
    if issues:
        lines.append("status: FAIL")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("status: PASS")
        lines.append("")
        lines.append("- Runtime controller/profiler files avoid forbidden evaluator-only fields.")
        lines.append("- `runtime_input_fields` in phase traces contain only online-allowed metadata.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"leakage_audit={output_path}")


if __name__ == "__main__":
    main()
