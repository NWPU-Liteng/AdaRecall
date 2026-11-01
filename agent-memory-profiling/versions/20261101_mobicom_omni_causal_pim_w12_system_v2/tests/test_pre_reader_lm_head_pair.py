import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_pre_reader_lm_head_pair import summarize


def run(mode, service, ttft, tokens):
    return {
        "complete": True,
        "evaluation_label_files_read": [],
        "rows": [{"question_id": "q", "service_seconds": service, "ttft_seconds": ttft,
                   "answer_token_ids": tokens, "answer": "answer"}],
        "lm_head_functional_dispatch": {
            "mode": mode, "physical_pim_executed": False,
            "calls": 1 if mode == "replace" else 0,
            "replaced_calls": 1 if mode == "replace" else 0,
            "resident_bytes": 128,
        },
    }


class PreReaderLmHeadPairTests(unittest.TestCase):
    def test_pair_gate_rejects_regression_and_accepts_equal_tokens(self):
        result = summarize(run("off", 10.0, 8.0, [1, 2]),
                           run("replace", 10.2, 8.1, [1, 2]))
        self.assertFalse(result["gate"]["passed"])
        self.assertTrue(result["aggregate"]["all_answer_tokens_equal"])


if __name__ == "__main__":
    unittest.main()
