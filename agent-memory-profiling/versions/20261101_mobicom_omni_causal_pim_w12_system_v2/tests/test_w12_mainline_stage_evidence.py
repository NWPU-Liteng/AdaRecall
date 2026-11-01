import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def stage(bank_calls):
    return {
        "bank_calls": bank_calls,
        "decode_fallback_calls": 0,
        "prefill_fallback_calls": 3,
        "fallback_rows": 7,
        "selected_modules": 2,
        "unique_layouts": 2,
        "layout_instance_sha256": "layout-1",
        "layout_identity_unchanged": True,
    }


def run_row(question_id, *, tokens, demand_text, object_ids):
    return {
        "question_id": question_id,
        "question_sha256": "question-" + question_id,
        "answer": "answer-" + question_id,
        "answer_token_ids": tokens,
        "proof_plan": {
            "query_parse": {
                "status": "ok",
                "output_token_count": 12,
                "parsed": {"source_facts": [{"demand_id": "d0", "text": demand_text}]},
            },
            "demands": [{"demand_id": "d0", "text": demand_text}],
            "grounding": {
                "status": "complete",
                "output_token_count": 8,
                "witness_count": 1,
                "parsed": {"witnesses": [{"demand_id": "d0"}]},
            },
            "selected_records": [{"object_ids": object_ids}],
        },
        "w12_stage_dispatch": {
            "query_parser": stage(2),
            "source_grounding": stage(3),
            "reader_generation": stage(4),
        },
    }


class W12MainlineStageEvidenceTests(unittest.TestCase):
    def write_run(self, path, row, reader_quantization="bf16", *, bank=False):
        document = {
            "complete": True,
            "reader_quantization": reader_quantization,
            "w12_numeric_mode": "bf16_reference",
            "rows": [row],
        }
        if bank:
            document["reader_w12_bank_materialization"] = {
                "contract_row_chunk": 4096,
            }
        if reader_quantization == "w12a16" and not bank:
            document["w12_host_contract_rows"] = 4096
        path.write_text(json.dumps(document))

    def test_audit_separates_execution_from_semantic_path_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            w12 = root / "w12.json"
            host = root / "host.json"
            output = root / "audit.json"
            self.write_run(w12, run_row(
                "q0", tokens=[1, 2, 3], demand_text="changed", object_ids=["b"]))
            self.write_run(host, run_row(
                "q0", tokens=[1, 2, 3], demand_text="original", object_ids=["a"]))
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/audit_w12_mainline_stage_evidence.py"),
                 "--w12", str(w12), "--host", str(host), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertTrue(result["gates"]["execution"])
            self.assertTrue(result["gates"]["decode_fallback_zero"])
            self.assertTrue(result["gates"]["single_layout"])
            self.assertFalse(result["gates"]["parser_exact"])
            self.assertFalse(result["gates"]["selected_evidence_exact"])
            self.assertEqual(result["summary"]["answer_equal"], 1)
            self.assertEqual(result["host_reader_quantization"], "bf16")
            self.assertFalse(result["physical_pim_executed"])

    def test_audit_identifies_w12_host_contract_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            w12 = root / "w12.json"
            host = root / "host_w12.json"
            output = root / "audit.json"
            row = run_row("q0", tokens=[1, 2, 3], demand_text="same",
                          object_ids=["a"])
            self.write_run(w12, row, reader_quantization="w12a16", bank=True)
            self.write_run(host, row, reader_quantization="w12a16")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/audit_w12_mainline_stage_evidence.py"),
                 "--w12", str(w12), "--host", str(host), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["host_reader_quantization"], "w12a16")
            self.assertEqual(result["w12_numeric_mode"], "bf16_reference")
            self.assertTrue(result["gates"]["contract_aligned"])
            self.assertTrue(any("isolates" in item for item in result["boundary"]))

    def test_audit_derives_bank_row_contract_from_layout_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            w12 = root / "w12.json"
            host = root / "host_w12.json"
            output = root / "audit.json"
            row = run_row("q0", tokens=[1, 2, 3], demand_text="same",
                          object_ids=["a"])
            self.write_run(w12, row, reader_quantization="w12a16", bank=False)
            w12_document = json.loads(w12.read_text())
            w12_document.update({"w12_bank_count": 256, "w12_tile_rows": 16})
            w12.write_text(json.dumps(w12_document))
            self.write_run(host, row, reader_quantization="w12a16")
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts/audit_w12_mainline_stage_evidence.py"),
                 "--w12", str(w12), "--host", str(host), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["bank_contract_rows"], 4096)
            self.assertTrue(result["gates"]["contract_aligned"])


if __name__ == "__main__":
    unittest.main()
