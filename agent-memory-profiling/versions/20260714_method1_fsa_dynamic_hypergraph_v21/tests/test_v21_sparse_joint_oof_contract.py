from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_r3_run_and_grid_are_immutable_contracts() -> None:
    config = json.loads((VERSION_ROOT / "configs" / "v21_sparse_joint_oof_r2.json").read_text())
    amendment = json.loads((VERSION_ROOT / "configs" / "v21_method1_closure_r3.json").read_text())
    assert config["run_id"] == amendment["sparse_joint_run_id"] == "v21_sparse_joint_oof_r2_20260714"
    assert config["activated_prototypes_grid"] == [4, 8, 16]
    assert config["members_per_semantic_edge_grid"] == [8, 16, 32]
    assert config["bundle_beam_width_grid"] == [8, 16, 32]
    assert config["final_candidate_budgets"] == [8, 16, 32]
    assert config["protected_ann_seeds"] == amendment["candidate_frontier_amendment"]["protected_ann_seeds"]
    assert config["calibration_read"] is False and config["formal_test_read"] is False


def test_worker_contains_r3_sparse_and_latency_guards() -> None:
    source = (VERSION_ROOT / "scripts" / "115_train_v21_sparse_joint_oof.py").read_text()
    for required in (
        "SparseSemanticLookupRequest", "construct_sparse_frontier", "semantic_bundle_unique_members",
        "fallback_members_appended", "budget_infeasible", "dense_online_scan", "OnlineLatencyRecorder",
        "rerank_executed", "embedding_precomputed",
    ):
        assert required in source
    assert "full eligible-by-prototype" not in source.lower()


def test_launcher_is_non_overwriting() -> None:
    source = (VERSION_ROOT / "scripts" / "116_launch_v21_sparse_joint_oof.py").read_text()
    assert "existing_attempt_requires_new_attempt" in source
    assert "write_json_new" in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
