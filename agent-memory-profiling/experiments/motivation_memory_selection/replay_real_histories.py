#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize replay-ready real-history manifest rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(manifest_path)
    counter = Counter(row.get("event_type", "unknown") for row in rows)
    report_path = output_dir / "replay_manifest_report.md"
    lines = [
        "# Replay Manifest Report",
        "",
        f"- manifest: `{manifest_path}`",
        f"- total_events: `{len(rows)}`",
    ]
    for key, value in sorted(counter.items()):
        lines.append(f"- {key}: `{value}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"replay_manifest_report={report_path}")


if __name__ == "__main__":
    main()
