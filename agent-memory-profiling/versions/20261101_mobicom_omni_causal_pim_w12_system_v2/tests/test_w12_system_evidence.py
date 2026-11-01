import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value))


class W12SystemEvidenceTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        projection = {
            "gpu_nonlinear_profile": {"median_ms_per_token": 5.0},
            "pim_grammar_model": {"milliseconds_per_token": 0.1},
            "matrix_linear_cost": {"host_linear_ms_per_token": 40.0},
            "target_ms_per_token": {"structured": 30.0,
                                    "free_text_reader": 29.9},
            "decode_token_accounting": {
                "decoder_bank_residency": {"matrices": [
                    {"name": "a", "out_features": 64, "in_features": 128},
                    {"name": "b", "out_features": 32, "in_features": 128},
                ]},
            },
            "requests_detail": [{
                "person_id": "0", "question_id": "q0",
                "baseline_service_seconds": 10.0, "baseline_ttft_seconds": 9.0,
                "parse_tokens": 2, "grounding_tokens": 3, "reader_tokens": 1,
                "control_service_seconds": 5.0, "control_ttft_seconds": 4.0,
                "target_service_seconds": 3.0, "target_ttft_seconds": 2.5,
                "parse_projection": {}, "grounding_projection": {},
            }],
        }
        timeline = {
            "arms": {
                "w12a16": {
                    "pim_linear_ms_per_token": 10.0,
                    "structured_ms_per_token": 19.1 * 1.15,
                    "reader_ms_per_token": 19.0 * 1.15,
                    "dispatch_ms_per_token": 4.0,
                    "totals": {
                        "refresh_cycles": 100,
                        "layer_fence_cycles": 20,
                        "external_transfer_bytes": 1000,
                        "weight_bytes": 2000,
                    },
                },
                "w12a16_joint_worst": {
                    "pim_linear_ms_per_token": 20.0,
                    "structured_ms_per_token": 29.1 * 1.15,
                    "reader_ms_per_token": 29.0 * 1.15,
                    "dispatch_ms_per_token": 4.0,
                    "totals": {"refresh_cycles": 200,
                               "layer_fence_cycles": 40,
                               "external_transfer_bytes": 1000,
                               "weight_bytes": 2000},
                },
                "w8a16_r2": {
                    "pim_linear_ms_per_token": 7.0,
                    "structured_ms_per_token": 16.1 * 1.15,
                    "reader_ms_per_token": 16.0 * 1.15,
                    "dispatch_ms_per_token": 4.0,
                    "totals": {},
                },
            }
        }
        capacity = {
            "variants": {"w12a16": {
                "decoder_plan": {"capacity_bytes": 100000,
                                 "total_bytes": 40000},
                "pool_with_bf16_embedding_alias_bytes": 45000,
            }}
        }
        raw = root / "raw"
        raw.mkdir()
        write_json(raw / "user_0.json", {"rows": [{
            "person_id": "0", "question_id": "q0",
            "prefill_seconds": 1.0, "decode_seconds": 2.0,
            "ttft_seconds": 9.0, "service_seconds": 10.0,
            "stop_reason": "eos",
        }]})
        projection_path = root / "projection.json"
        timeline_path = root / "timeline.json"
        capacity_path = root / "capacity.json"
        write_json(projection_path, projection)
        write_json(timeline_path, timeline)
        write_json(capacity_path, capacity)
        return projection_path, timeline_path, capacity_path, raw

    def run_script(self, name, *arguments):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
            cwd=ROOT, capture_output=True, text=True)
        if completed.returncode:
            self.fail(completed.stdout + completed.stderr)
        return completed

    def test_request_ledger_and_ablation_are_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection, timeline, capacity, raw = self.fixture(directory)
            ledger = root / "ledger.json"
            self.run_script(
                "build_w12_request_ledger.py", "--projection", projection,
                "--timeline", timeline, "--raw-root", raw, "--output", ledger)
            result = json.loads(ledger.read_text())
            self.assertEqual(result["requests"], 1)
            self.assertEqual(result["decode_inventory"]["decode_tokens"], 6)
            self.assertEqual(len(result["request_ledger"][0]["stage_events"]), 3)
            self.assertGreater(
                result["decode_inventory"]["external_reduction_vs_host_bf16"], 1)

            ablations = root / "ablations.json"
            self.run_script(
                "project_w12_system_ablations.py", "--projection", projection,
                "--timeline", timeline, "--capacity-audit", capacity,
                "--output", ablations)
            result = json.loads(ablations.read_text())
            self.assertTrue(result["capacity"]["copies"]["single_shared_decoder"]["feasible"])
            self.assertFalse(result["capacity"]["copies"]["three_stage_copies"]["feasible"])
            self.assertGreater(
                result["arms"]["w12_full_decoder_shared"]["speedup_vs_host_control"]["ttft_sum"],
                result["arms"]["w12_reader_only_pim"]["speedup_vs_host_control"]["ttft_sum"])
            self.assertIn(
                "cross_stage_reuse_saved_seconds_per_cold_start",
                result["cold_load"])
            self.assertNotIn(
                "cross_stage_reuse_saved_seconds_per_request",
                result["cold_load"])
            self.assertIn("not once per request", result["cold_load"]["amortization"])

    def test_device_profile_parameter_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projection, timeline, capacity, raw = self.fixture(directory)
            ledger = root / "ledger.json"
            self.run_script(
                "build_w12_request_ledger.py", "--projection", projection,
                "--timeline", timeline, "--raw-root", raw, "--output", ledger)
            profile = json.loads(
                (ROOT / "configs/edge_lpddr_pim_w12_target_v1.json").read_text())
            profile["config"]["bank_count"] = 32
            profile_path = root / "profile.json"
            write_json(profile_path, profile)
            output = root / "calibration.json"
            completed = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts/audit_w12_device_calibration.py"),
                 "--timeline", str(timeline), "--request-ledger", str(ledger),
                 "--capacity-audit", str(capacity), "--profile", str(profile_path),
                 "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("timeline parameters differ", completed.stderr)


if __name__ == "__main__":
    unittest.main()
