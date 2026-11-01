import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_adaptive_frontier_pair import audit_pair


def row(question_id, service, ttft, *, answer_tokens=(1, 2), seed=2, selected=2):
    return {
        'question_id': question_id,
        'question_sha256': 'query-' + question_id,
        'answer_sha256': 'answer-' + question_id,
        'answer_token_ids': list(answer_tokens),
        'stop_reason': 'eos',
        'input_tokens': 100,
        'service_seconds': service,
        'ttft_seconds': ttft,
        'prefill_seconds': 1.,
        'decode_seconds': 1.,
        'execution_events': [
            {'phase': 'source_grounding', 'seconds': service / 2},
        ],
        'grounding_frontier': {
            'state_count': 8,
            'packed_bytes': 800,
        },
        'grounding_frontier_levels': [
            {'level': 0, 'candidate_count': seed, 'packed_bytes': 200,
             'lost_source_provenance_count': 1,
             'lost_source_relation_count': 0,
             'lost_binding_key_count': 0},
            {'level': 1, 'candidate_count': selected, 'packed_bytes': 400,
             'lost_source_provenance_count': 0,
             'lost_source_relation_count': 0,
             'lost_binding_key_count': 0},
        ],
        'proof_plan': {'grounding': {
            'frontier_selected_level': 1,
            'frontier_attempts': [{'level': 0}, {'level': 1}],
        }},
    }


def write_run(path, rows):
    path.write_text(json.dumps({'complete': True, 'evaluation_label_files_read': [],
                                'rows': rows}))


def write_protocol_run(path, rows, *, query_contract_enabled=False):
    path.write_text(json.dumps({
        'complete': True,
        'evaluation_label_files_read': [],
        'prepared_audit_sha256': 'prepared',
        'cohort_sha256': 'cohort',
        'index_sha256': 'index',
        'reader_model': 'reader',
        'embedder_model': 'embedder',
        'reader_processor_input_budget': 8192,
        'total_text_tokens': 3600,
        'max_images': 4,
        'max_new_tokens': 512,
        'query_contract_enabled': query_contract_enabled,
        'rows': rows,
    }))


class AdaptiveFrontierPairTests(unittest.TestCase):
    def test_pair_passes_only_with_exact_outputs_and_e2e_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            candidate = root / 'candidate.json'
            write_run(reference, [row('q0', 10., 5.)])
            write_run(candidate, [row('q0', 8., 4.)])
            result = audit_pair(reference, candidate, min_requests=1)
        self.assertEqual(result['promotion_decision'], 'pass')
        self.assertEqual(result['quality_gate'], 'pass')
        self.assertEqual(result['latency_gate'], 'pass')
        self.assertGreater(result['measured_e2e_speedup'], 1.05)

    def test_quality_mismatch_rejects_even_when_candidate_is_faster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            candidate = root / 'candidate.json'
            write_run(reference, [row('q0', 10., 5.)])
            changed = row('q0', 5., 2., answer_tokens=(9,))
            changed['answer_sha256'] = 'changed'
            write_run(candidate, [changed])
            result = audit_pair(reference, candidate, min_requests=1)
        self.assertEqual(result['promotion_decision'], 'reject')
        self.assertEqual(result['quality_gate'], 'failed_exact_output_contract')
        self.assertAlmostEqual(result['ttft_p50_speedup'], 2.5)
        self.assertEqual(result['pair_mismatches'][0]['mismatches'],
                         ['answer_token_ids', 'answer_sha256'])

    def test_pair_requires_same_request_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            candidate = root / 'candidate.json'
            write_run(reference, [row('q0', 10., 5.)])
            write_run(candidate, [row('q1', 8., 4.)])
            with self.assertRaises(ValueError):
                audit_pair(reference, candidate)

    def test_pair_rejects_protocol_mismatch_before_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'reference.json'
            candidate = root / 'candidate.json'
            write_protocol_run(reference, [row('q0', 10., 5.)])
            write_protocol_run(candidate, [row('q0', 8., 4.)], query_contract_enabled=True)
            result = audit_pair(reference, candidate, min_requests=1)
        self.assertEqual(result['protocol_gate'], 'failed_protocol_mismatch')
        self.assertEqual(result['protocol_mismatches'], ['query_contract_enabled'])
        self.assertEqual(result['promotion_decision'], 'reject')


if __name__ == '__main__':
    unittest.main()
