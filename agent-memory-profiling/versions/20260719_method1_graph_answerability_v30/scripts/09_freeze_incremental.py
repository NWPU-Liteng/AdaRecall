from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import yaml

from common import REPOSITORY_ROOT, VERSION_ROOT, V28_ROOT, sha256_file, write_json


def output(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("tracked worktree changes must be committed before incremental freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("staged changes must be committed before incremental freeze")

    config_path = VERSION_ROOT / "configs" / "incremental_gate_g.yaml"
    test_path = VERSION_ROOT / "artifacts" / "frozen" / "incremental_test_receipt_attempt_002.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if not test["passed"]:
        raise RuntimeError("passing incremental test receipt required")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frozen_inputs = config["frozen_inputs"]
    frontier = V28_ROOT / "runs" / frozen_inputs["frontier_run_id"]
    layout = VERSION_ROOT / "runs" / frozen_inputs["layout_run_id"]
    graph_config = V28_ROOT / "configs" / "graph_only_main.yaml"
    expected = {
        "graph_config_sha256": frozen_inputs["graph_config_sha256"],
        "frontier_rows_sha256": frozen_inputs["frontier_rows_sha256"],
        "frontier_snapshot_manifest_sha256": frozen_inputs["frontier_snapshot_manifest_sha256"],
        "layout_rows_sha256": frozen_inputs["layout_rows_sha256"],
        "layout_aggregate_sha256": frozen_inputs["layout_aggregate_sha256"],
        "layout_snapshot_manifest_sha256": frozen_inputs["layout_snapshot_manifest_sha256"],
    }
    observed = {
        "graph_config_sha256": sha256_file(graph_config),
        "frontier_rows_sha256": sha256_file(frontier / "per_query_frontier.jsonl"),
        "frontier_snapshot_manifest_sha256": sha256_file(frontier / "causal_snapshot_manifest.json"),
        "layout_rows_sha256": sha256_file(layout / "per_query_layout_io_e2e.jsonl"),
        "layout_aggregate_sha256": sha256_file(layout / "aggregate_layout_io_e2e.json"),
        "layout_snapshot_manifest_sha256": sha256_file(layout / "snapshot_build_manifest.json"),
    }
    if observed != expected:
        raise RuntimeError(f"incremental input lineage mismatch: {observed}")
    source_paths = [
        path
        for folder in ("src", "scripts", "tests", "configs")
        for path in (VERSION_ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    manifest = {
        "schema_version": "method1-v30-incremental-source-manifest-attempt-002-1",
        "files": [
            {"path": str(path.relative_to(VERSION_ROOT)), "sha256": sha256_file(path)}
            for path in sorted(source_paths)
        ],
    }
    frozen = VERSION_ROOT / "artifacts" / "frozen"
    manifest_path = frozen / "incremental_source_manifest_attempt_002.json"
    receipt_path = frozen / "incremental_freeze_receipt_attempt_002.json"
    write_json(manifest_path, manifest, exclusive=True)
    receipt = {
        "schema_version": "method1-v30-incremental-freeze-attempt-002-1",
        "status": "FROZEN_BEFORE_INCREMENTAL_GATE_G_ATTEMPT_002",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "test_receipt_sha256": sha256_file(test_path),
        **observed,
        "incremental_runs_allowed": 1,
        "compaction_mode": "full_exact_boundary_rebuild",
        "local_compaction_claim": False,
        "native_agent_access_claim": False,
        "fresh_authorized": False,
    }
    write_json(receipt_path, receipt, exclusive=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
