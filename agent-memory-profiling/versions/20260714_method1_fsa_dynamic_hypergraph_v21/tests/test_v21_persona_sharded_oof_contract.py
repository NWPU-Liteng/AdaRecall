from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_r2_sharded_contract_has_no_replication_or_effect_threshold() -> None:
    config = json.loads((VERSION_ROOT / "configs/v21_persona_sharded_hyper_index_oof_r2.json").read_text())
    assert config["feature_replication_factor"] == 0
    assert config["rank_metadata_tile_bytes"] == 4096
    assert config["minimum_effect_threshold"] is None
    assert config["calibration_read"] is False
    assert config["formal_test_read"] is False


def test_aggregator_requires_exact_member_equivalence() -> None:
    source = (VERSION_ROOT / "scripts/138_aggregate_v21_persona_sharded_hyper_index_oof.py").read_text()
    assert "returned_members_identical_all_queries" in source
    assert "old_and_corrected_tile_totals_directly_comparable" in source
    assert "minimum_effect_threshold" in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
