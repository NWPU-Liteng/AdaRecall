import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from omni_prepare import ImageResolver, build_graph, normalize_time, prepare, source_event_candidates


class OmniPrepareTests(unittest.TestCase):
    def test_timestamp_repair_is_explicit_and_parseable(self):
        normalized, rule = normalize_time("2025-08:05 09:25:00")
        self.assertEqual(normalized, "2025-08-05 09:25:00")
        self.assertEqual(rule, "month_day_separator")

    def record(self):
        return {
            "uuid": 0,
            "sessions": [
                {
                    "session_id": "0_0", "event_id": 0, "parent_event_id": None,
                    "event_start_time": "2025-01-01 08:00:00", "event_end_time": "2025-01-01 09:00:00",
                    "event_name": "root", "image_refs": {},
                    "dialogue": [{"role": "user", "content_type": "text", "content": "hello"}],
                },
                {
                    "session_id": "0_1_2", "event_id": "1_2", "parent_event_id": 0,
                    "event_start_time": "2025-01-02 08:00:00", "event_end_time": "2025-01-02 09:00:00",
                    "event_name": "child", "image_refs": {"event_scene": "image/uid0/a.png"},
                    "dialogue": [{"role": "user", "content_type": "image",
                                  "image_inline": "image/uid0/a.png"}],
                },
            ],
        }

    def test_graph_keeps_observable_dialogue_and_unresolved_parent(self):
        graph, identity = build_graph(self.record(), ImageResolver(None))
        self.assertEqual(len(graph["atoms"]), 2)
        self.assertEqual(graph["dependencies"], {})
        self.assertNotIn('parent_event_id', graph['sessions'][1])
        self.assertNotIn('event_name', graph['sessions'][0])
        self.assertIn("0_1_2", identity["session_ids"])
        self.assertEqual(graph["atoms"][1]["image_refs"][0]["status"], "missing")

    def test_integer_parent_event_resolves_child_sessions(self):
        graph, identity = build_graph(self.record(), ImageResolver(None))
        self.assertEqual(source_event_candidates(0, identity), ["0_0"])
        self.assertEqual(source_event_candidates(1, identity), [])
        self.assertEqual(source_event_candidates('1_2', identity), ["0_1_2"])
        self.assertEqual(source_event_candidates(9, identity), [])

    def test_prepare_separates_labels_and_declares_history_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            questions = root / "questions.jsonl"
            output = root / "prepared"
            data.write_text(json.dumps(self.record(), ensure_ascii=False) + "\n")
            questions.write_text(json.dumps({"uuid": 0, "questions": [{
                "question_id": "q0", "question": "what?", "answer": "answer",
                "source_session_ids": ["0_1_2"], "source_event_ids": ['1_2'],
                "image_refs": [], "question_type": "single_hop", "difficulty": "easy"}]},
                ensure_ascii=False) + "\n")
            audit = prepare(data, questions, output, None)
            manifest = json.loads((output / "manifests.jsonl").read_text())
            label = json.loads((output / "labels.jsonl").read_text())
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(manifest["visibility_protocol"], "history_end")
            self.assertEqual(manifest["effective_timestamp"], "2025-01-02 09:00:00")
            self.assertNotIn("answer", manifest)
            self.assertFalse(any('source' in key or 'image_refs' in key for key in manifest))
            self.assertEqual(label["golden_answers"], ["answer"])
            self.assertNotIn("golden_answers", json.loads((output / "omni_graph.json").read_text())[0])


if __name__ == "__main__":
    unittest.main()
