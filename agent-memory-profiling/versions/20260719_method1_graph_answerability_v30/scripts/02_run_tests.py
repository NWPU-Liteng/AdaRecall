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
        "schema_version": "method1-v30-qwen-control-tests-1",
        "status": "TESTS_PASSED" if completed.returncode == 0 else "TESTS_FAILED",
        "passed": completed.returncode == 0,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_file_sha256": sha256_file(VERSION_ROOT / "tests" / "test_qwen_packing.py"),
        "packer_source_sha256": sha256_file(VERSION_ROOT / "src" / "qwen_reranker.py"),
        "stdout": completed.stdout,
    }
    write_json(VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_test_receipt.json", receipt)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(json.dumps({"status": receipt["status"]}, indent=2))


if __name__ == "__main__":
    main()
