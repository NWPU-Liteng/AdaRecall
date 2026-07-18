from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from common import REPOSITORY_ROOT, VERSION_ROOT, V28_ROOT, sha256_file, write_json


def output(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("tracked worktree changes must be committed before system freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("staged changes must be committed before system freeze")
    config_path = VERSION_ROOT / "configs" / "system_gates.yaml"
    test_path = VERSION_ROOT / "artifacts" / "frozen" / "system_test_receipt.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    if not test["passed"]:
        raise RuntimeError("passing system test receipt required")
    graph_config = V28_ROOT / "configs" / "graph_only_main.yaml"
    frontier = V28_ROOT / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f"
    qwen = VERSION_ROOT / "runs" / "v30_corrected_qwen_oof_20260719"
    expected = {
        "graph_config_sha256": "d6252a2e669cb07226393d2bd6b90888f8973af8038dfaa4281b2b9174fb5e8b",
        "frontier_rows_sha256": "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98",
        "frontier_aggregate_sha256": "8a618b0b958f64597e4064996126e51fb50d0ee0bf3dbf3cde9fcfe1f0f7de99",
        "corrected_qwen_rows_sha256": "475fdb788cb4d2cf64880cafd3743f10b5f00045d89f61f6596780a076df0805",
        "corrected_qwen_aggregate_sha256": "ad29fd28ef585e0240876ad2f80b66ed1623be0e8ef0f6df18feffc1dba60c35",
    }
    observed = {
        "graph_config_sha256": sha256_file(graph_config),
        "frontier_rows_sha256": sha256_file(frontier / "per_query_frontier.jsonl"),
        "frontier_aggregate_sha256": sha256_file(frontier / "aggregate_metrics.json"),
        "corrected_qwen_rows_sha256": sha256_file(qwen / "per_query_qwen.jsonl"),
        "corrected_qwen_aggregate_sha256": sha256_file(qwen / "aggregate_qwen.json"),
    }
    if observed != expected:
        raise RuntimeError(f"system input lineage mismatch: {observed}")
    source_paths = [
        path
        for folder in ("src", "scripts", "tests", "configs")
        for path in (VERSION_ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    manifest = {
        "schema_version": "method1-v30-system-source-manifest-1",
        "files": [
            {"path": str(path.relative_to(VERSION_ROOT)), "sha256": sha256_file(path)}
            for path in sorted(source_paths)
        ],
    }
    frozen = VERSION_ROOT / "artifacts" / "frozen"
    manifest_path = frozen / "system_source_manifest.json"
    receipt_path = frozen / "system_freeze_receipt.json"
    write_json(manifest_path, manifest, exclusive=True)
    receipt = {
        "schema_version": "method1-v30-system-freeze-1",
        "status": "FROZEN_BEFORE_LAYOUT_IO_E2E_INCREMENTAL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "test_receipt_sha256": sha256_file(test_path),
        **observed,
        "layout_io_runs_allowed": 1,
        "incremental_runs_allowed": 1,
        "fresh_authorized": False,
    }
    write_json(receipt_path, receipt, exclusive=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
