from __future__ import annotations

import json

from common import REPOSITORY_ROOT, VERSION_ROOT
from src.provenance import sha256_file


def main() -> None:
    manifest_path = VERSION_ROOT / "artifacts" / "v27_provenance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v27 = VERSION_ROOT.parent / "20260718_prism_aligned_method1_closure_v27"
    checks = {
        "config_sha256": v27 / "configs" / "v27.json",
        "source_sha256": v27 / "src" / "v27" / "core.py",
        "runner_sha256": v27 / "scripts" / "01_run_dual_objective_oof.py",
        "rows_sha256": v27 / "runs" / "v27_oof" / "relevance_rows.jsonl",
        "layout_rows_sha256": v27 / "runs" / "v27_oof" / "layout_rows.jsonl",
        "aggregate_sha256": v27 / "aggregate" / "dual_objective_oof.json",
    }
    for key, path in checks.items():
        actual = sha256_file(path)
        if actual != manifest["current"][key]:
            raise RuntimeError(f"v27 audit input changed: {key} expected {manifest['current'][key]}, got {actual}")
    if manifest["status"] != "PROVENANCE_CONFLICT" or manifest["promotion_allowed"]:
        raise RuntimeError("v27 conflict quarantine was weakened")
    print("V27_PROVENANCE_CONFLICT_CONFIRMED")


if __name__ == "__main__":
    main()
