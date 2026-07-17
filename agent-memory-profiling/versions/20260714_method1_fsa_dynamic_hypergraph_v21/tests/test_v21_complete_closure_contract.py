from __future__ import annotations

from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]


def test_complete_closure_keeps_claim_boundaries() -> None:
    source = (VERSION_ROOT / "scripts" / "133_write_v21_method1_complete_closure.py").read_text()
    assert '"minimum_effect_threshold": None' in source
    assert '"agent_answer_accuracy_evaluated": False' in source
    assert '"true_causal_dynamic_stream_evaluated": False' in source
    assert '"pim_hardware_evaluated": False' in source
    assert '"rerank_executed": False' in source
    assert '"formal_test_read": False' in source


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
