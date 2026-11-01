import copy
import hashlib
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_smoke_closure import audit_performance_pair, audit_records, empirical_latency, verify_source_snapshot


class OmniSmokeClosureTests(unittest.TestCase):
    def fixture(self):
        prediction_hash = hashlib.sha256(b'answer').hexdigest()
        prompt_hash = hashlib.sha256(b'prompt').hexdigest()
        smoke = {'complete': True, 'evaluation_label_files_read': [],
                 'source_files_sha256': {'reader.py': 'source'}, 'performance_comparable': True,
                 'operator_trace': {'functional_samples': {}},
                 'rows': [{'question_id': 'q', 'answer': 'answer', 'answer_sha256': prediction_hash,
                           'generation_equivalent': True, 'stop_reason': 'eos',
                           'images': [{'archive_member': 'image.png'}], 'retrieved_session_ids': ['s'],
                           'service_seconds': 2.0, 'prefill_seconds': 1.0, 'decode_seconds': 0.5,
                           'input_tokens': 123}]}
        requests = {'judge_requests': [{'question_id': 'q', 'prediction_sha256': prediction_hash,
                                       'prompt': 'prompt', 'prompt_sha256': prompt_hash}]}
        judge = {'complete': True, 'rows': [{'question_id': 'q', 'prediction_sha256': prediction_hash,
                 'prompt_sha256': prompt_hash, 'parse_valid': True, 'value': 1.0}]}
        projection = {'resident_bytes': 10, 'capacity_bytes': 20,
                      'warm_linear_speedup_scenario': 1.3, 'cold_linear_speedup_scenario': 1.2,
                      'end_to_end_speedup': None, 'full_model_residency_validated': False}
        labels = {'q': {'source_session_ids': ['s'], 'evidence_image_refs': []}}
        registration = {'question_ids': ['q'], 'source_files_sha256': {'reader.py': 'source'}}
        return smoke, requests, judge, projection, labels, registration

    def test_success_is_only_execution_not_mainline_qualification(self):
        result = audit_records(*self.fixture())
        self.assertEqual(result['execution_integrity'], 'pass')
        self.assertEqual(result['local_semantic_proxy'], 1.0)
        self.assertFalse(result['main_experiment_closed'])
        self.assertFalse(result['latency_statistics_qualified'])
        self.assertIsNone(result['rows'][0]['label_image_reference_recall'])

    def test_truncation_and_invalid_judgment_stay_in_strict_denominator(self):
        smoke, requests, judge, projection, labels, registration = self.fixture()
        smoke['rows'][0]['stop_reason'] = 'max_new_tokens'
        smoke['rows'][0]['generation_equivalent'] = None
        result = audit_records(smoke, requests, judge, projection, labels, registration,
                               require_verified_execution=False)
        self.assertEqual(result['local_semantic_proxy'], 1.0)
        self.assertEqual(result['strict_local_proxy_success_rate'], 0.0)
        self.assertEqual(result['attempted_requests'], 1)
        self.assertEqual(result['truncated_question_ids'], ['q'])
        self.assertFalse(result['generation_equivalence_qualified'])
        judge['rows'][0].update(parse_valid=False, value=None)
        result = audit_records(smoke, requests, judge, projection, labels, registration,
                               require_verified_execution=False)
        self.assertEqual(result['invalid_judgments'], 1)
        self.assertEqual(result['strict_local_proxy_success_rate'], 0.0)

    def test_quantiles_use_ceil_rank_and_reject_nonfinite_data(self):
        result = empirical_latency(list(range(1, 17)))
        self.assertEqual(result['p50_nearest_rank_seconds'], 8)
        self.assertEqual(result['p95_nearest_rank_seconds'], 16)
        for values in ([], [float('nan')], [-1], [float('inf')]):
            with self.assertRaises(ValueError):
                empirical_latency(values)

    def test_frozen_snapshot_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.py'
            source.write_bytes(b'original')
            smoke = {'source_files_sha256': {'source.py': hashlib.sha256(b'original').hexdigest()}}
            registration = {'source_snapshot_directory': directory}
            self.assertTrue(verify_source_snapshot(smoke, registration))
            source.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_source_snapshot(smoke, registration)

    def test_mutated_incomplete_duplicate_and_false_acceleration_fail(self):
        for kind in ('prediction', 'prompt', 'source', 'truncated', 'duplicate', 'capacity', 'e2e', 'label'):
            smoke, requests, judge, projection, labels, registration = copy.deepcopy(self.fixture())
            if kind == 'prediction':
                smoke['rows'][0]['answer'] = 'changed'
            elif kind == 'prompt':
                requests['judge_requests'][0]['prompt'] = 'changed'
            elif kind == 'source':
                registration['source_files_sha256'] = {}
            elif kind == 'truncated':
                smoke['rows'][0]['stop_reason'] = 'max_new_tokens'
            elif kind == 'duplicate':
                judge['rows'] *= 2
            elif kind == 'capacity':
                projection['resident_bytes'] = 21
            elif kind == 'e2e':
                projection['end_to_end_speedup'] = 1.3
            else:
                smoke['evaluation_label_files_read'] = ['labels.jsonl']
            with self.assertRaises(ValueError):
                audit_records(smoke, requests, judge, projection, labels, registration)

    def test_hyperedge_context_and_executed_images_must_match(self):
        smoke, *others = self.fixture()
        smoke['rows'][0]['hypergraph_evidence_plan'] = {
            'selected_hyperedges': [{'atom_ids': ['a']}], 'accepted_atom_ids': ['a'],
            'text_tokens': 5, 'text_budget': 10, 'max_images': 1,
            'images': [{'normalized': 'different.png'}]}
        smoke['rows'][0]['images'][0]['source_atom_id'] = 'a'
        with self.assertRaises(ValueError):
            audit_records(smoke, *others)

    def test_performance_pair_requires_exact_inputs_outputs_and_no_validation(self):
        verified = {'complete': True, 'performance_comparable': True,
                    'numerical_probe_overhead_in_service': False, 'verify_generation': True}
        protocol_names = ('reader_config_sha256', 'embedder_config_sha256', 'prepared_audit_sha256',
                          'archive_sha256', 'system_prompt_sha256', 'evidence_policy',
                          'candidate_sessions', 'max_images', 'max_new_tokens', 'total_text_tokens')
        verified.update({name: 'fixed' for name in protocol_names})
        verified['rows'] = [{name: 'fixed' for name in (
            'question_id', 'question_sha256', 'answer_sha256', 'answer_token_ids',
            'input_tokens', 'images', 'retrieved_session_ids', 'evidence_selection',
            'hypergraph_evidence_plan', 'stop_reason')}]
        verified['rows'][0]['generation_equivalent'] = True
        verified['rows'][0].update({name: 1 for name in (
            'service_seconds', 'prefill_seconds', 'decode_seconds', 'retrieval_seconds',
            'service_peak_cuda_allocated_bytes', 'service_peak_cuda_reserved_bytes')})
        performance = copy.deepcopy(verified)
        performance['verify_generation'] = False
        self.assertEqual(audit_performance_pair(verified, performance)['input_output_pair_integrity'], 'pass')
        for kind in ('answer', 'evidence', 'validation', 'probe', 'protocol'):
            changed = copy.deepcopy(performance)
            if kind == 'answer':
                changed['rows'][0]['answer_token_ids'] = ['different']
            elif kind == 'evidence':
                changed['rows'][0]['hypergraph_evidence_plan'] = 'different'
            elif kind == 'validation':
                changed['verify_generation'] = True
            elif kind == 'probe':
                changed['numerical_probe_overhead_in_service'] = True
            else:
                changed['reader_config_sha256'] = 'different'
            with self.assertRaises(ValueError):
                audit_performance_pair(verified, changed)


if __name__ == '__main__':
    unittest.main()
