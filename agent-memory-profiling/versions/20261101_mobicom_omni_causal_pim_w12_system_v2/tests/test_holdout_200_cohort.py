import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_holdout_200_cohort import select


class HoldoutCohortTests(unittest.TestCase):
    def test_selects_only_sealed_users_and_writes_no_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            manifests = prepared / 'manifests.jsonl'
            rows = []
            for user, split in (('6', 'holdout'), ('7', 'holdout'), ('0', 'development')):
                for index in range(5):
                    rows.append({'question_id': f'{user}_q_{index}', 'person_id': user,
                                 'split': split, 'effective_timestamp': f'2025-01-0{index + 1} 00:00:00',
                                 'question_index': index, 'answer': 'secret'})
            manifests.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            split = {'holdout_people': ['6', '7']}
            (prepared / 'split.json').write_text(json.dumps(split))
            audit = {'artifacts_sha256': {'manifests.jsonl': hashlib.sha256(
                manifests.read_bytes()).hexdigest()}}
            (prepared / 'audit.json').write_text(json.dumps(audit))
            result = select(prepared, ['6', '7'], 3, 'test-namespace')
            self.assertEqual(result['requests'], 6)
            self.assertEqual(result['serving_splits'], ['holdout'])
            self.assertFalse(result['evaluation_label_files_read'])
            self.assertTrue(all(len(ids) == 3 for ids in result['question_ids_by_user'].values()))
            self.assertNotIn('secret', json.dumps(result))
            with self.assertRaises(ValueError):
                select(prepared, ['0'], 3, 'test-namespace')


if __name__ == '__main__':
    unittest.main()
