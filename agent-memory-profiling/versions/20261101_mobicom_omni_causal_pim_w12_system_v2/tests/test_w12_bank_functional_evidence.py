import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def dispatch(mode, *, observations=4, argmax_matches=4):
    return {
        "mode": mode,
        "numeric_mode": "fp32_bank",
        "selected_modules": 2,
        "bank_calls": 4,
        "decode_fallback_calls": 0,
        "prefill_fallback_calls": 2,
        "fallback_rows": 8,
        "bank_external_bytes": 100,
        "fallback_external_bytes": 200,
        "resident_bytes": 1000,
        "released_source_parameter_bytes": 2000,
        "source_released": mode == "replace",
        "all_selected_modules_executed": True,
        "silent_fallback": False,
        "shadow": {
            "observations": observations if mode == "shadow" else 0,
            "argmax_matches": argmax_matches if mode == "shadow" else 0,
            "max_absolute_error": 0.1 if mode == "shadow" else None,
            "max_relative_l2_error": 0.001 if mode == "shadow" else None,
        },
        "physical_pim_executed": False,
        "performance_comparable": False,
    }


def write_run(path, mode, rows):
    path.write_text(json.dumps({
        "complete": True,
        "rows": rows,
        "reader_w12_bank_dispatch": dispatch(mode),
    }))


def row(question_id, tokens):
    return {
        "question_id": question_id,
        "question_sha256": "question-" + question_id,
        "answer_token_ids": tokens,
        "answer_sha256": "answer-" + question_id + "-" + "-".join(map(str, tokens)),
    }


class W12BankFunctionalEvidenceTests(unittest.TestCase):
    def test_audit_separates_execution_envelope_and_prefix_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shadow = root / "shadow.json"
            replace = root / "replace.json"
            host = root / "host.json"
            output = root / "audit.json"
            write_run(shadow, "shadow", [
                row("q0", [1, 2, 3]),
                row("q1", [4, 5, 6]),
            ])
            write_run(replace, "replace", [
                row("q0", [1, 2, 3]),
                row("q1", [4, 9, 6]),
            ])
            write_run(host, "replace", [
                row("q0", [1, 2, 3, 7]),
                row("q1", [4, 5, 6]),
            ])
            completed = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts/audit_w12_bank_functional_evidence.py"),
                 "--shadow", str(shadow), "--replace", str(replace),
                 "--host-contract", str(host), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["gates"]["execution"])
            self.assertTrue(result["gates"]["numeric_envelope"])
            self.assertEqual(result["gates"]["semantic_quality"],
                             "not_assessed_by_prefix_audit")
            self.assertEqual(result["prefix_comparison"]["first_token_matches"], 2)
            self.assertEqual(
                result["prefix_comparison"]["full_generated_prefix_matches"], 1)


if __name__ == "__main__":
    unittest.main()
