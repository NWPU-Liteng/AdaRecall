from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=REPO).returncode:
        raise RuntimeError("commit v29 before freeze")
    receipt = ROOT / "artifacts" / "freeze_receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError("v29 freeze receipt already exists")
    value = {
        "schema_version": "method1-v29-freeze-1",
        "status": "FROZEN_BEFORE_CROSSFIT_OOF",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config_sha256": sha(ROOT / "configs" / "posterior.json"),
        "posterior_source_sha256": sha(ROOT / "src" / "posterior.py"),
        "crossfit_source_sha256": sha(ROOT / "scripts" / "01_run_crossfit_oof.py"),
        "v28_frontier_rows_sha256": "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98",
        "v28_qwen_rows_sha256": "439d32655eed0d6e136a181c9b66d373ad54e13937fbe16fe8131eea9e559696"
    }
    receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
