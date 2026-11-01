import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from build_main_cohort import build
from evaluate_main_experiment import evaluate
from omni_proof_runtime import authorize_users


class MainExperimentContractTests(unittest.TestCase):
    def test_cohort_does_not_copy_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            prepared.mkdir()
            manifests = prepared / 'manifests.jsonl'
            manifests.write_text(json.dumps({
                'question_id': '0_q_0', 'person_id': '0', 'split': 'holdout',
                'effective_timestamp': '2025-01-01 00:00:00', 'question': 'Where?'
            }) + '\n')
            import hashlib
            manifest_hash = hashlib.sha256(manifests.read_bytes()).hexdigest()
            split = prepared / 'split.json'
            split.write_text(json.dumps({'development_people': [], 'validation_people': [],
                                         'holdout_people': ['0']}))
            audit = prepared / 'audit.json'
            audit.write_text(json.dumps({'artifacts_sha256': {
                'manifests.jsonl': manifest_hash, 'split.json': hashlib.sha256(split.read_bytes()).hexdigest()
            }}))
            output = root / 'cohort.json'
            payload = build(prepared, output, {'holdout'})
            self.assertEqual(payload['questions'], 1)
            self.assertNotIn('question', json.loads(output.read_text()))
            self.assertEqual(payload['evaluation_label_files_read'], [])

    def test_all_split_authorization_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            prepared.mkdir()
            split = prepared / 'split.json'
            split.write_text(json.dumps({'development_people': ['0'], 'validation_people': ['1'],
                                         'holdout_people': ['2']}))
            import hashlib
            (prepared / 'audit.json').write_text(json.dumps({'artifacts_sha256': {
                'split.json': hashlib.sha256(split.read_bytes()).hexdigest()
            }}))
            authorize_users(prepared, ['2'], allowed_splits={'holdout'})
            with self.assertRaises(ValueError):
                authorize_users(prepared, ['2'], allowed_splits={'development'})

    def test_evaluation_reports_missing_judge_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            run_root = root / 'run'
            prepared.mkdir()
            run_root.mkdir()
            (prepared / 'audit.json').write_text('{}')
            (prepared / 'labels.jsonl').write_text(json.dumps({
                'question_id': '0_q_0', 'person_id': '0', 'golden_answers': ['Monday'],
                'question_type': 'single_hop'
            }) + '\n')
            (run_root / 'user_0_shard_0.json').write_text(json.dumps({
                'complete': True, 'rows': [{'question_id': '0_q_0', 'answer': 'Monday'}]
            }))
            result = evaluate(prepared, run_root, root / 'evaluation.json', root / 'Qwen3-8B')
            self.assertEqual(result['judge']['status'], 'missing_requested_qwen3_8b')
            self.assertFalse(result['closure'])


if __name__ == '__main__':
    unittest.main()
