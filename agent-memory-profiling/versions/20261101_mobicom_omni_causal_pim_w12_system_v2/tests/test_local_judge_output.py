import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from judge_mobilemem_local import reserve_output


class LocalJudgeOutputTests(unittest.TestCase):
    def test_exclusive_reservation_precedes_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'judge.json'
            reserve_output(path)
            original = path.read_bytes()
            self.assertFalse(json.loads(original)['complete'])
            with self.assertRaises(ValueError):
                reserve_output(path)
            self.assertEqual(path.read_bytes(), original)

    def test_existing_completed_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'judge.json'
            original = b'{"complete": true, "rows": []}'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                reserve_output(path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
