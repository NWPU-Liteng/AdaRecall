from __future__ import annotations

from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_r2_closure_preserves_claim_boundaries() -> None:
    source = (VERSION_ROOT / "scripts/147_write_v21_method1_complete_closure_r2.py").read_text()
    assert '"minimum_effect_threshold": None' in source
    assert '"agent_answer_execution": "not_evaluated"' in source
    assert '"persona_sharded_incremental_publication": "not_yet_integrated"' in source
    assert '"causal_dynamic_access_update": "not_eligible_from_source_data"' in source
    assert '"formal_test_read": False' in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
