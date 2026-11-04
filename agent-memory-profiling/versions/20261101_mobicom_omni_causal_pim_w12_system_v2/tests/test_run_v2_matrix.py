import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_v2_matrix import wait_for_judge_complete


class FakeJudgeProcess:
    def __init__(self, *, exit_code=None):
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code

    def wait(self):
        return self.exit_code or 0


class RunV2MatrixTests(unittest.TestCase):
    def test_waits_for_complete_marker_even_if_process_exited(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            (run_root / 'judge_complete.json').write_text(json.dumps({
                'complete': True,
                'questions': 56,
            }))
            wait_for_judge_complete(FakeJudgeProcess(exit_code=0), run_root, 56)

    def test_rejects_process_exit_without_complete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'before completion marker'):
                wait_for_judge_complete(
                    FakeJudgeProcess(exit_code=0), Path(directory), 56)


if __name__ == '__main__':
    unittest.main()
