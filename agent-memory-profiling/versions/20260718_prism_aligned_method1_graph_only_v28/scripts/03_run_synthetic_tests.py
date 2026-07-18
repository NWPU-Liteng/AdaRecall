from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from common import VERSION_ROOT, write_json


def main() -> None:
    command = ["python", "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=VERSION_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    receipt = {
        "schema_version": "method1-v28-synthetic-test-receipt-1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": completed.returncode,
        "output": completed.stdout,
        "passed": completed.returncode == 0,
    }
    write_json(VERSION_ROOT / "artifacts" / "frozen" / "synthetic_test_receipt.json", receipt)
    print(completed.stdout, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
