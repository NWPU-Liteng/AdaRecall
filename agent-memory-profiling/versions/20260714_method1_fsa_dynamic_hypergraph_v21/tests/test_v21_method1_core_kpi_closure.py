from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_core_kpi_revision_has_no_effect_threshold() -> None:
    config = json.loads((VERSION_ROOT / "configs" / "v21_method1_core_kpi_closure_r1.json").read_text())
    assert config["minimum_effect_threshold"] is None
    assert config["threshold_based_method_failure"] is False
    assert config["calibration_read"] is False
    assert config["formal_test_read"] is False
    assert config["deployment_point"] == {
        "activated_prototypes": 16,
        "members_per_semantic_edge": 32,
        "semantic_bundle_beam": 32,
        "candidate_pool_cap": 96,
        "budgets": [8, 16, 32],
    }


def test_core_kpi_script_defines_both_recall_semantics() -> None:
    source = (VERSION_ROOT / "scripts" / "131_close_v21_method1_core_kpis.py").read_text()
    assert "group_recall_at_k" in source
    assert "evidence_member_recall_at_k" in source
    assert "exact_task_success_rate" in source
    assert "minimum_effect_threshold" in source
    assert "formal_test_read" in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
