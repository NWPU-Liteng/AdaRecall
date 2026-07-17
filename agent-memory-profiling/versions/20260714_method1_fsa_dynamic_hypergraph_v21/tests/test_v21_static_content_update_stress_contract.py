from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_static_update_stress_is_noncausal_and_locked() -> None:
    config = json.loads((VERSION_ROOT / "configs" / "v21_static_content_update_stress_r1.json").read_text())
    assert config["growth_points_percent_of_full_corpus"] == [1, 5, 10, 20]
    assert config["minimum_effect_threshold"] is None
    assert config["causal_stream_claim"] is False
    assert config["access_order_claim"] is False
    assert config["co_access_claim"] is False
    assert config["pim_hardware_claim"] is False
    assert config["calibration_read"] is False
    assert config["formal_test_read"] is False


def test_static_update_script_requires_exact_lookup_equivalence() -> None:
    source = (VERSION_ROOT / "scripts" / "132_run_v21_static_content_update_stress.py").read_text()
    assert "lookup_mismatch_count" in source
    assert "overlay/compacted lookup mismatch" in source
    assert "chronological_rank" in source
    assert "ordering_role" in source
    assert "snapshot_build_ms" in source
    assert "atomic_publish_ms" in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
