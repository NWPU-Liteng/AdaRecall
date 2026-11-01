import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_persistent_session_replay import build_replay


class PersistentSessionReplayTests(unittest.TestCase):
    def record(self):
        return {
            "uuid": 0,
            "sessions": [
                {
                    "session_id": "0_0", "event_id": 0,
                    "event_start_time": "2025-01-01 08:00:00",
                    "event_end_time": "2025-01-01 09:00:00",
                    "image_refs": {},
                    "dialogue": [{"role": "user", "content": "first"}],
                },
                {
                    "session_id": "0_1", "event_id": 1,
                    "event_start_time": "2025-01-02 08:00:00",
                    "event_end_time": "2025-01-02 09:00:00",
                    "image_refs": {},
                    "dialogue": [{"role": "user", "content": "second"}],
                },
            ],
        }

    def test_checkpoint_exposes_source_but_not_future_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text(json.dumps(self.record()) + "\n")
            questions = {
                "uuid": 0,
                "questions": [
                    {
                        "question_id": "q0", "question": "what happened?",
                        "answer": "hidden", "source_session_ids": ["0_0"],
                    },
                ],
            }
            (root / "questions.jsonl").write_text(json.dumps(questions) + "\n")
            output = root / "replay"
            result = build_replay(root / "data.jsonl", root / "questions.jsonl", output)
            self.assertEqual(result["queries"], 1)
            query = json.loads((output / "queries.jsonl").read_text())
            self.assertEqual(query["visible_session_ids"], ["0_0"])
            self.assertNotIn("answer", query)
            self.assertNotIn("source_session_ids", query)
            audit = json.loads((output / "schedule_audit.jsonl").read_text())
            self.assertEqual(audit["resolved_source_session_ids"], ["0_0"])
            self.assertTrue(audit["temporal_replay_eligible"])

    def test_missing_source_uses_explicit_history_end_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.jsonl").write_text(json.dumps(self.record()) + "\n")
            questions = {
                "uuid": 0,
                "questions": [{
                    "question_id": "q0", "question": "unknown?",
                    "answer": "hidden", "source_session_ids": [],
                }],
            }
            (root / "questions.jsonl").write_text(json.dumps(questions) + "\n")
            output = root / "replay"
            build_replay(root / "data.jsonl", root / "questions.jsonl", output)
            query = json.loads((output / "queries.jsonl").read_text())
            self.assertEqual(query["checkpoint_reason"], "history_end_no_source_session")
            self.assertEqual(query["visible_session_ids"], ["0_0", "0_1"])
            audit = json.loads((output / "schedule_audit.jsonl").read_text())
            self.assertFalse(audit["temporal_replay_eligible"])


if __name__ == "__main__":
    unittest.main()
