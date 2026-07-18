from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.provenance import create_immutable_run


ROOT = Path(__file__).resolve().parents[1]


def test_v27_conflict_is_quarantined() -> None:
    manifest = json.loads((ROOT / "artifacts" / "v27_provenance_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PROVENANCE_CONFLICT"
    assert manifest["promotion_allowed"] is False
    assert manifest["current"]["aggregate_sha256"].startswith("09203e19")
    assert manifest["handoff"]["recorded_aggregate_sha256"].startswith("51f3a05e")


def test_run_directories_are_immutable(tmp_path: Path) -> None:
    create_immutable_run(tmp_path, "formal_001")
    with pytest.raises(FileExistsError):
        create_immutable_run(tmp_path, "formal_001")
