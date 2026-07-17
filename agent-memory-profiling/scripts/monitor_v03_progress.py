#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_LAYOUT_NAMES = [
    "random_balanced",
    "write_time_order",
    "semantic_balanced_cluster",
    "pairwise_graph_greedy",
    "global_hypergraph_greedy",
    "global_hypergraph_semantic_tiebreak",
]
CONTEXT_LAYOUT_NAME = "context_hypergraph_greedy"
REPORT_FILES = [
    "layout_replay_summary.csv",
    "online_frontier_summary.csv",
    "adaptive_frontier_summary.md",
    "phase_timings.md",
    "README_e1.md",
]


@dataclass
class ProgressSnapshot:
    output_dir: Path
    bank_count: int
    capacities: list[int]
    expected_base_layouts: int
    expected_context_metadata: int
    expected_shared_metadata: int
    mapping_count: int
    base_layout_metadata_count: int
    context_metadata_count: int
    shared_metadata_count: int
    report_status: dict[str, tuple[bool, int]]
    latest_files: list[tuple[float, Path]]
    active_processes: list[str]
    run_manifest: dict | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor progress for the Method1 v0.3 formal experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "versions" / "v0.3" / "formal_run",
        help="Experiment output directory to inspect.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Omit or use 0 for a single snapshot.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=8,
        help="How many recently touched files to show.",
    )
    return parser.parse_args()


def load_config(output_dir: Path) -> dict:
    config_path = output_dir / "configs_snapshot" / "method1_v0_3_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config snapshot: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def list_recent_files(root: Path, limit: int) -> list[tuple[float, Path]]:
    if not root.exists():
        return []
    candidates = [path for path in root.rglob("*") if path.is_file()]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [(path.stat().st_mtime, path) for path in candidates[:limit]]


def discover_processes(output_dir: Path) -> list[str]:
    cmd = [
        "ps",
        "-eo",
        "pid,etime,%cpu,%mem,cmd",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return [f"ps failed: {exc}"]

    output_dir_str = str(output_dir)
    matched: list[str] = []
    fallback: list[str] = []
    for line in result.stdout.splitlines():
        if "run_method1_v03.py" not in line:
            continue
        fallback.append(line.strip())
        if output_dir_str in line or "--output-dir versions/v0.3/formal_run" in line:
            matched.append(line.strip())
    return matched or fallback


def load_run_manifest(output_dir: Path) -> dict | None:
    manifest_path = output_dir / "artifacts" / "run_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def take_snapshot(output_dir: Path, tail: int) -> ProgressSnapshot:
    config = load_config(output_dir)
    manifest_path = output_dir / "manifests" / "bank_causal_split.jsonl"
    bank_count = count_jsonl_rows(manifest_path) if manifest_path.exists() else 0
    capacities = [int(capacity) for capacity in config.get("capacities", [])]
    layouts_root = output_dir / "layouts"

    mapping_count = sum(1 for _ in layouts_root.rglob("memory_to_tile.jsonl"))
    base_layout_metadata_count = 0
    context_metadata_count = 0
    for path in layouts_root.rglob("layout_metadata.json"):
        layout_name = path.parent.parent.name
        if layout_name == CONTEXT_LAYOUT_NAME:
            context_metadata_count += 1
        else:
            base_layout_metadata_count += 1
    shared_metadata_count = sum(1 for _ in layouts_root.rglob("structure_metadata.json"))

    reports_dir = output_dir / "reports"
    report_status: dict[str, tuple[bool, int]] = {}
    for name in REPORT_FILES:
        path = reports_dir / name if name != "README_e1.md" else output_dir / name
        report_status[name] = (path.exists(), path.stat().st_size if path.exists() else 0)

    recent_roots = [layouts_root, reports_dir, output_dir / "artifacts"]
    latest_files: list[tuple[float, Path]] = []
    for root in recent_roots:
        latest_files.extend(list_recent_files(root, tail))
    latest_files.sort(key=lambda item: item[0], reverse=True)
    latest_files = latest_files[:tail]

    expected_base_layouts = bank_count * len(capacities) * len(BASE_LAYOUT_NAMES)
    expected_context_metadata = bank_count * len(capacities)
    expected_shared_metadata = bank_count * len(capacities)

    return ProgressSnapshot(
        output_dir=output_dir,
        bank_count=bank_count,
        capacities=capacities,
        expected_base_layouts=expected_base_layouts,
        expected_context_metadata=expected_context_metadata,
        expected_shared_metadata=expected_shared_metadata,
        mapping_count=mapping_count,
        base_layout_metadata_count=base_layout_metadata_count,
        context_metadata_count=context_metadata_count,
        shared_metadata_count=shared_metadata_count,
        report_status=report_status,
        latest_files=latest_files,
        active_processes=discover_processes(output_dir),
        run_manifest=load_run_manifest(output_dir),
    )


def format_pct(done: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(100.0 * done / total):6.2f}%"


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_snapshot(snapshot: ProgressSnapshot) -> str:
    lines = []
    lines.append("Method1 v0.3 Formal Run Monitor")
    lines.append(f"output_dir: {snapshot.output_dir}")
    lines.append(
        f"banks: {snapshot.bank_count} | capacities: {snapshot.capacities} | "
        f"expected_base_layouts: {snapshot.expected_base_layouts}"
    )
    lines.append("")
    lines.append("Layout progress")
    lines.append(
        f"- base mappings: {snapshot.mapping_count}/{snapshot.expected_base_layouts} "
        f"({format_pct(snapshot.mapping_count, snapshot.expected_base_layouts)})"
    )
    lines.append(
        f"- base layout metadata: {snapshot.base_layout_metadata_count}/{snapshot.expected_base_layouts} "
        f"({format_pct(snapshot.base_layout_metadata_count, snapshot.expected_base_layouts)})"
    )
    lines.append(
        f"- context blocker metadata: {snapshot.context_metadata_count}/{snapshot.expected_context_metadata} "
        f"({format_pct(snapshot.context_metadata_count, snapshot.expected_context_metadata)})"
    )
    lines.append(
        f"- shared structure metadata: {snapshot.shared_metadata_count}/{snapshot.expected_shared_metadata} "
        f"({format_pct(snapshot.shared_metadata_count, snapshot.expected_shared_metadata)})"
    )
    lines.append("")
    lines.append("Reports")
    for name in REPORT_FILES:
        exists, size = snapshot.report_status[name]
        state = "ready" if exists else "pending"
        lines.append(f"- {name}: {state} (bytes={size})")
    lines.append("")
    lines.append("Host process")
    if snapshot.active_processes:
        for line in snapshot.active_processes:
            lines.append(f"- {line}")
    else:
        lines.append("- no matching run_method1_v03.py process found")
    lines.append("")
    lines.append("Recent writes")
    if snapshot.latest_files:
        for mtime, path in snapshot.latest_files:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            lines.append(f"- {stamp} {path}")
    else:
        lines.append("- no files found under layouts/reports/artifacts")
    lines.append("")
    lines.append("Run manifest")
    if snapshot.run_manifest is None:
        lines.append("- artifacts/run_manifest.json not present yet")
    else:
        elapsed = snapshot.run_manifest.get("elapsed_seconds")
        device = snapshot.run_manifest.get("device")
        cache_stats = snapshot.run_manifest.get("layout_cache_stats")
        lines.append(f"- elapsed_seconds: {elapsed}")
        lines.append(f"- device: {device}")
        lines.append(f"- layout_cache_stats: {cache_stats}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        print(f"output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    if args.watch and args.watch > 0:
        try:
            while True:
                snapshot = take_snapshot(output_dir, tail=args.tail)
                clear_screen()
                print(render_snapshot(snapshot))
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0

    snapshot = take_snapshot(output_dir, tail=args.tail)
    print(render_snapshot(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
