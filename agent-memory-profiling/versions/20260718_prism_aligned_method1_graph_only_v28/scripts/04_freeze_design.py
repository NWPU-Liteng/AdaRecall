from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

from common import REPOSITORY_ROOT, VERSION_ROOT, write_json
from src.provenance import sha256_file, source_manifest


def output(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("tracked worktree changes must be committed before design freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY_ROOT).returncode != 0:
        raise RuntimeError("staged changes must be committed before design freeze")
    test_receipt_path = VERSION_ROOT / "artifacts" / "frozen" / "synthetic_test_receipt.json"
    if not test_receipt_path.is_file() or not json.loads(test_receipt_path.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("passing synthetic test receipt required")
    config_path = VERSION_ROOT / "configs" / "graph_only_main.yaml"
    frozen = VERSION_ROOT / "artifacts" / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    frozen_config = frozen / "frozen_graph_only_config.yaml"
    if frozen_config.exists():
        raise FileExistsError("design freeze already exists")
    shutil.copyfile(config_path, frozen_config)
    paths = [
        path
        for folder in ("src", "scripts", "tests", "configs")
        for path in (VERSION_ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    manifest = source_manifest(paths, base=VERSION_ROOT)
    write_json(frozen / "source_manifest.json", manifest)
    receipt = {
        "schema_version": "method1-v28-design-freeze-1",
        "status": "FROZEN_BEFORE_FORMAL_OOF",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "git_branch": output(["git", "branch", "--show-current"]),
        "config_sha256": sha256_file(config_path),
        "frozen_config_sha256": sha256_file(frozen_config),
        "source_manifest_sha256": sha256_file(frozen / "source_manifest.json"),
        "synthetic_test_receipt_sha256": sha256_file(test_receipt_path),
        "formal_oof_runs_allowed": 1,
        "formal_oof_runs_consumed": 0,
        "forbidden": ["MedMemoryBench development", "MedMemoryBench test", "consumed LoCoMo"],
    }
    write_json(frozen / "design_freeze_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
