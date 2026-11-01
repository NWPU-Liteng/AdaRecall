import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_verified_subset import audit_subset


class VerifiedSubsetTests(unittest.TestCase):
    def fixture(self):
        names = ('reader_config_sha256', 'embedder_config_sha256', 'prepared_audit_sha256',
                 'archive_sha256', 'system_prompt_sha256', 'evidence_policy', 'candidate_sessions',
                 'max_images', 'max_new_tokens', 'total_text_tokens')
        verified = {name: 'fixed' for name in names}
        row = {name: 'fixed' for name in ('question_sha256', 'answer_sha256', 'answer_token_ids',
               'input_tokens', 'images', 'retrieved_session_ids', 'evidence_selection',
               'hypergraph_evidence_plan', 'stop_reason')}
        row.update(question_id='q1', generation_equivalent=True)
        row.update({name: 1 for name in ('service_seconds', 'prefill_seconds', 'decode_seconds',
                   'retrieval_seconds', 'service_peak_cuda_allocated_bytes', 'service_peak_cuda_reserved_bytes')})
        verified.update(complete=True, rows=[row], operator_trace={'functional_samples': {
            name: {'allclose': True} for name in ('gate', 'down', 'head')}})
        performance = copy.deepcopy(verified)
        performance.update(performance_comparable=True, verify_generation=False,
                           numerical_probe_overhead_in_service=False)
        performance['rows'].append({**row, 'question_id': 'q2'})
        return verified, performance

    def test_subset_never_qualifies_unverified_requests(self):
        result = audit_subset(*self.fixture())
        self.assertEqual(result['verified_question_ids'], ['q1'])
        self.assertFalse(result['full_performance_stream_generation_equivalence_qualified'])
        verified, performance = self.fixture()
        performance['rows'].reverse()
        with self.assertRaises(ValueError):
            audit_subset(verified, performance)
        verified, performance = self.fixture()
        verified['operator_trace']['functional_samples'].pop('head')
        with self.assertRaises(ValueError):
            audit_subset(verified, performance)


if __name__ == '__main__':
    unittest.main()
