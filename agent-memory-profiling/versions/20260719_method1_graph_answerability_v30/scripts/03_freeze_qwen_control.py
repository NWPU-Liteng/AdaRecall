from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from common import REPOSITORY_ROOT, VERSION_ROOT, V28_ROOT, sha256_file, write_json


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("tracked worktree changes must be committed before freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("staged changes must be committed before freeze")
    config_path = VERSION_ROOT / "configs" / "qwen_suffix_preserving.yaml"
    test_path = VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_test_receipt.json"
    test_receipt = json.loads(test_path.read_text(encoding="utf-8"))
    if not test_receipt["passed"]:
        raise RuntimeError("passing Qwen packing test receipt required")
    cuda_smoke_path = VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_cuda_smoke.json"
    cuda_smoke = json.loads(cuda_smoke_path.read_text(encoding="utf-8"))
    if not cuda_smoke["passed"]:
        raise RuntimeError("passing corrected-Qwen CUDA smoke required")
    frontier_dir = V28_ROOT / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f"
    expected = {
        "frontier_rows_sha256": "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98",
        "frontier_aggregate_sha256": "8a618b0b958f64597e4064996126e51fb50d0ee0bf3dbf3cde9fcfe1f0f7de99",
    }
    observed = {
        "frontier_rows_sha256": sha256_file(frontier_dir / "per_query_frontier.jsonl"),
        "frontier_aggregate_sha256": sha256_file(frontier_dir / "aggregate_metrics.json"),
    }
    if observed != expected:
        raise RuntimeError(f"frozen v28 Frontier lineage mismatch: {observed}")
    source_paths = [
        path
        for folder in ("src", "scripts", "tests", "configs")
        for path in (VERSION_ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    source_manifest = {
        "schema_version": "method1-v30-qwen-control-source-manifest-1",
        "files": [
            {"path": str(path.relative_to(VERSION_ROOT)), "sha256": sha256_file(path)}
            for path in sorted(source_paths)
        ],
    }
    frozen = VERSION_ROOT / "artifacts" / "frozen"
    manifest_path = frozen / "qwen_control_source_manifest.json"
    receipt_path = frozen / "qwen_control_freeze_receipt.json"
    write_json(manifest_path, source_manifest, exclusive=True)
    receipt = {
        "schema_version": "method1-v30-qwen-control-freeze-1",
        "status": "FROZEN_BEFORE_CANDIDATE_IDENTICAL_QWEN_OOF",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "test_receipt_sha256": sha256_file(test_path),
        "cuda_smoke_sha256": sha256_file(cuda_smoke_path),
        **observed,
        "formal_runs_allowed": 1,
        "formal_runs_consumed": 0,
        "candidate_ids_frozen": True,
        "identical_packing_for_all_methods": True,
        "fresh_authorized": False,
    }
    write_json(receipt_path, receipt, exclusive=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
