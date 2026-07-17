#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Verify tenant isolation from V4 phase traces.")
    parser.add_argument("--phase-trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.phase_trace).resolve())
    violations = [row for row in rows if int(row.get("cross_tenant_candidate_count", 0)) != 0]
    output_path = Path(args.output).resolve()
    lines = ["# Tenant Isolation Report", ""]
    if violations:
        lines.append("status: FAIL")
        lines.append("")
        for row in violations[:50]:
            lines.append(f"- event `{row.get('event_id')}` phase `{row.get('phase')}` cross_tenant_candidate_count={row.get('cross_tenant_candidate_count')}")
    else:
        lines.append("status: PASS")
        lines.append("")
        lines.append(f"- checked_rows: `{len(rows)}`")
        lines.append("- all `cross_tenant_candidate_count` values are zero")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"tenant_isolation_report={output_path}")


if __name__ == "__main__":
    main()
