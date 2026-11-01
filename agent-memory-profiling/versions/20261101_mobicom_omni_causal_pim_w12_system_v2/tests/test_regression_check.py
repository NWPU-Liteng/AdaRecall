import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from regression_check import compare


def _row(answer='same', candidates=('a', 'b')):
    return {
        'question_id': '0_q_1',
        'person_id': '0',
        'answer': answer,
        'proof_plan': {
            'accepted_object_ids': ['text:1'],
            'proof_status': 'ok',
            'selection_mode': 'mode',
            'query_contract': {'status': 'ok'},
        },
        'grounding_frontier': {
            'records': [
                {'demand_id': 'd0', 'base_record_id': record_id}
                for record_id in candidates
            ],
        },
    }


def _write_run(path: Path, row: dict) -> None:
    path.mkdir()
    (path / 'user_0_shard_0.json').write_text(json.dumps({
        'complete': True,
        'person_id': '0',
        'rows': [row],
    }))


class RegressionCheckTests(unittest.TestCase):
    def test_identical_runs_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root / 'a', _row())
            _write_run(root / 'b', _row())
            failures, reported = compare(root / 'a', root / 'b')
            self.assertEqual(failures, [])
            self.assertEqual(reported, [])

    def test_candidate_order_difference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root / 'a', _row(candidates=('a', 'b')))
            _write_run(root / 'b', _row(candidates=('b', 'a')))
            failures, _ = compare(root / 'a', root / 'b')
            self.assertEqual(len(failures), 1)
            self.assertIn('demand_candidates differs', failures[0])

    def test_declared_nondeterministic_field_only_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root / 'a', _row(answer='first'))
            _write_run(root / 'b', _row(answer='second'))
            (root / 'NONDETERMINISTIC_FIELDS.json').write_text(json.dumps({
                'fields': ['answer'],
            }))
            failures, reported = compare(root / 'a', root / 'b')
            self.assertEqual(failures, [])
            self.assertEqual(len(reported), 1)


if __name__ == '__main__':
    unittest.main()
