from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_r2_latency_revision_is_new_and_locked() -> None:
    config = json.loads((VERSION_ROOT / "configs" / "v21_sparse_joint_tuned_oof_r2.json").read_text())
    assert config["run_id"] == "v21_sparse_joint_tuned_oof_r2_20260714"
    assert config["parent_run_id"] == "v21_sparse_joint_tuned_oof_r1_20260714"
    assert config["candidate_pool_cap"] == 96
    assert config["selection_budget_cap"] == 32
    assert config["calibration_read"] is False
    assert config["formal_test_read"] is False


def test_r2_worker_stops_selector_at_budget_cap() -> None:
    source = (VERSION_ROOT / "scripts" / "128_train_v21_sparse_joint_tuned_oof_r2.py").read_text()
    assert "target_budget" in source
    assert "max_budget=max(budgets)" in source
    assert "no_dense_online_incidence_scan" in source
    assert '"rerank_executed": False' in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
