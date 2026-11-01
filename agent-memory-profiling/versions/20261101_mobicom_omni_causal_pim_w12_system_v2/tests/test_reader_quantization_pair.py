import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_reader_quantization_pair as pairing


class ReaderQuantizationPairTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        cohort = {
            'complete': True, 'prepared_audit_sha256': 'prepared',
            'evaluation_label_files_read': [], 'runtime_labels_read': False,
            'question_ids_by_user': {'0': ['q0', 'q1']},
        }
        cohort_path = root / 'cohort.json'
        cohort_path.write_text(json.dumps(cohort))
        runs = {}
        for arm in ('bf16', 'w8a8'):
            run_root = root / arm
            run_root.mkdir()
            (run_root / 'cohort.json').write_bytes(cohort_path.read_bytes())
            run = {'complete': True, 'evaluation_label_files_read': [],
                   'reader_quantization': arm, 'prepared_audit_sha256': 'prepared'}
            for field in pairing.PROTOCOL_FIELDS:
                run[field] = 'fixed'
            for field in pairing.IDENTITY_FIELDS:
                run[field] = {'source': 'fixed'}
            run['prepared_audit_sha256'] = 'prepared'
            run['rows'] = [{
                'question_id': question_id, 'person_id': '0',
                'answer': 'same', 'answer_sha256': 'same-hash',
                'answer_token_ids': [1, 2, 3], 'stop_reason': 'eos',
                'question_sha256': 'question', 'effective_timestamp': 'time',
                'visibility_protocol': 'history_end', 'input_tokens': 10,
                'visual_patches': 0, 'retrieved_session_ids': ['s0'],
                'images': [], 'evidence_selection': [{'session_id': 's0'}],
                'hypergraph_evidence_plan': {
                    'text': 'same',
                    'embedding_producer_costs': {
                        'lookup_seconds': 1.0, 'encode_seconds': 2.0, 'write_seconds': 0.0},
                    'embedding_producers': [
                        {'producer_sha256': 'same', 'lookup_seconds': 1.0, 'encode_seconds': 2.0}],
                },
                'embedding_producers': [{'producer_sha256': 'same'}],
                'processor_inputs_sha256': {'input_ids': 'same'},
            } for question_id in ('q0', 'q1')]
            (run_root / 'baseline_user_0.json').write_text(json.dumps(run))
            (run_root / 'baseline_user_0.registration.json').write_text(
                json.dumps({'question_ids': ['q0', 'q1']}))
            runs[arm] = run_root
        return cohort_path, runs

    def test_exact_pair_passes_and_differences_are_located(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, roots = self.fixture(directory)
            path = roots['w8a8'] / 'baseline_user_0.json'
            run = json.loads(path.read_text())
            run['rows'][0]['hypergraph_evidence_plan']['embedding_producer_costs'] = {
                'lookup_seconds': 99.0, 'encode_seconds': 88.0, 'write_seconds': 77.0}
            path.write_text(json.dumps(run))
            result = pairing.audit(roots['bf16'], roots['w8a8'], cohort)
            self.assertTrue(result['gate']['passed'])
            self.assertEqual(result['generation']['complete_greedy_token_matches'], 2)

            candidate_path = roots['w8a8'] / 'baseline_user_0.json'
            candidate = json.loads(candidate_path.read_text())
            candidate['rows'][0]['answer_token_ids'] = [1, 9, 3]
            candidate_path.write_text(json.dumps(candidate))
            result = pairing.audit(roots['bf16'], roots['w8a8'], cohort)
            self.assertFalse(result['gate']['passed'])
            self.assertEqual(result['generation']['complete_greedy_token_matches'], 1)
            alignment = result['rows'][0]['token_alignment']
            self.assertEqual(alignment['common_prefix_tokens'], 1)
            self.assertEqual(alignment['reference_token_at_divergence'], 2)
            self.assertEqual(alignment['candidate_token_at_divergence'], 9)

    def test_non_quantization_and_pre_reader_differences_fail(self):
        mutations = ('quantization', 'label', 'protocol', 'identity', 'upstream',
                     'cohort', 'registration')
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                cohort, roots = self.fixture(directory)
                path = roots['w8a8'] / 'baseline_user_0.json'
                run = json.loads(path.read_text())
                if mutation == 'quantization':
                    run['reader_quantization'] = 'bf16'
                elif mutation == 'label':
                    run['evaluation_label_files_read'] = ['labels.jsonl']
                elif mutation == 'protocol':
                    run['max_new_tokens'] = 999
                elif mutation == 'identity':
                    run['source_files_sha256'] = {'source': 'changed'}
                elif mutation == 'upstream':
                    run['rows'][0]['evidence_selection'] = [{'session_id': 'other'}]
                path.write_text(json.dumps(run))
                if mutation == 'cohort':
                    (roots['w8a8'] / 'cohort.json').write_text('{}')
                elif mutation == 'registration':
                    (roots['w8a8'] / 'baseline_user_0.registration.json').write_text(
                        json.dumps({'question_ids': ['q1', 'q0']}))
                with self.assertRaises(ValueError):
                    pairing.audit(roots['bf16'], roots['w8a8'], cohort)

    def test_quantization_source_allowlist_is_explicit_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, roots = self.fixture(directory)
            path = roots['w8a8'] / 'baseline_user_0.json'
            run = json.loads(path.read_text())
            allowed = sorted(pairing.QUANTIZATION_SOURCE_ALLOWLIST)[0]
            run['source_files_sha256'] = {
                'source': 'fixed', allowed: 'quantization-only-change'}
            reference_path = roots['bf16'] / 'baseline_user_0.json'
            reference = json.loads(reference_path.read_text())
            reference['source_files_sha256'] = {
                'source': 'fixed', allowed: 'reference'}
            reference_path.write_text(json.dumps(reference))
            path.write_text(json.dumps(run))
            result = pairing.audit(roots['bf16'], roots['w8a8'], cohort)
            self.assertEqual(
                result['integrity']['quantization_source_differences_by_user'],
                {'0': [allowed]})

            run['source_files_sha256']['other.py'] = 'changed'
            reference['source_files_sha256']['other.py'] = 'reference'
            reference_path.write_text(json.dumps(reference))
            path.write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                pairing.audit(roots['bf16'], roots['w8a8'], cohort)

    def test_stop_mismatch_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort, roots = self.fixture(directory)
            path = roots['w8a8'] / 'baseline_user_0.json'
            run = json.loads(path.read_text())
            run['rows'][1]['stop_reason'] = 'max_new_tokens'
            path.write_text(json.dumps(run))
            result = pairing.audit(roots['bf16'], roots['w8a8'], cohort)
            self.assertEqual(result['generation']['stop_reason_matches'], 1)
            self.assertFalse(result['gate']['passed'])


if __name__ == '__main__':
    unittest.main()
