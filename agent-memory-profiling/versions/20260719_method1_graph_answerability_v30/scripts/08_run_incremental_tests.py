from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

from common import VERSION_ROOT, sha256_file, write_json


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(VERSION_ROOT / "tests")],
        cwd=VERSION_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    receipt = {
        "schema_version": "method1-v30-incremental-tests-attempt-002-1",
        "status": "TESTS_PASSED" if completed.returncode == 0 else "TESTS_FAILED",
        "passed": completed.returncode == 0,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout": completed.stdout,
        "incremental_test_sha256": sha256_file(VERSION_ROOT / "tests" / "test_incremental_gate_g.py"),
        "system_test_sha256": sha256_file(VERSION_ROOT / "tests" / "test_system_contracts.py"),
        "incremental_source_sha256": sha256_file(VERSION_ROOT / "src" / "incremental_update.py"),
        "layout_source_sha256": sha256_file(VERSION_ROOT / "src" / "causal_tile_layout.py"),
        "runner_source_sha256": sha256_file(VERSION_ROOT / "scripts" / "10_run_incremental_agent.py"),
        "config_sha256": sha256_file(VERSION_ROOT / "configs" / "incremental_gate_g.yaml"),
    }
    write_json(
        VERSION_ROOT / "artifacts" / "frozen" / "incremental_test_receipt_attempt_002.json",
        receipt,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(json.dumps({"status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
