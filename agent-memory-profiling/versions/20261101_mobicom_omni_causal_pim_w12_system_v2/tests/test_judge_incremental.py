import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from judge_incremental import ledger_identities


class JudgeIncrementalTests(unittest.TestCase):
    def test_expected_filter_excludes_old_development_ledgers(self):
        with tempfile.TemporaryDirectory() as directory:
            rows_root = Path(directory)
            for user, question in (('0', '0_q_dev'), ('4', '4_q_val')):
                path = rows_root / user / f'{question}.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'question_id': question}))
            self.assertEqual(
                ledger_identities(rows_root, {'4_q_val'}),
                {'4_q_val'})


if __name__ == '__main__':
    unittest.main()
