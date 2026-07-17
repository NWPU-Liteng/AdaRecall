from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_tuned_revision_is_new_and_locked() -> None:
    config = json.loads((VERSION_ROOT / "configs" / "v21_sparse_joint_tuned_oof_r1.json").read_text())
    assert config["run_id"] == "v21_sparse_joint_tuned_oof_r1_20260714"
    assert config["parent_run_id"] == "v21_sparse_joint_oof_r2_20260714"
    assert config["candidate_pool_cap"] == 96
    assert config["candidate_pool_cap"] >= max(config["final_candidate_budgets"])
    assert config["frontier_shared_across_methods"] is True
    assert config["calibration_read"] is False
    assert config["formal_test_read"] is False


def test_tuned_worker_uses_bounded_shared_frontier() -> None:
    source = (VERSION_ROOT / "scripts" / "125_train_v21_sparse_joint_tuned_oof.py").read_text()
    assert 'load_config(args.config)' in source
    assert 'config["candidate_pool_cap"]' in source
    assert "construct_sparse_frontier" in source
    assert "candidate_frontier_shared_across_methods" in source
    assert "no_dense_online_incidence_scan" in source
    assert '"calibration_read": False' in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
