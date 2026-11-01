import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_evidence_prefilter_shadow import summarize


class EvidencePrefilterSummaryTests(unittest.TestCase):
    def test_summary_includes_baseline_frontier_and_host_coverage_gates(self):
        rows = [{
            'question_id': 'q0',
            'service_seconds': 12.0,
            'proof_plan': {'evidence_state_prefilter_shadow': {
                'seconds': 0.01,
                'input_candidate_count': 4,
                'filtered_candidate_count': 2,
                'candidate_reduction_fraction': 0.5,
                'frontier_audit': {
                    'baseline_frontier_candidate_count': 2,
                    'filtered_baseline_frontier_candidate_count': 1,
                    'baseline_frontier_reduction_fraction': 0.5,
                    'host_witness_count': 1,
                    'host_witness_missing_ids': [],
                    'host_witness_coverage': True,
                    'safe_to_replace_host_grounding': False,
                },
                'exact_metadata': True,
                'applied': True,
                'fallback_reason': None,
                'answer_path_unchanged': True,
                'coverage_gate_passed': True,
            }},
        }]
        result = summarize(rows)
        self.assertEqual(result['baseline_frontier_candidate_reduction_fraction']['mean'], 0.5)
        self.assertEqual(result['applied_host_witness_coverage_rate'], 1.0)
        self.assertEqual(result['coverage_gate_pass_rate'], 1.0)
        self.assertEqual(result['applied_requests_with_missing_host_witnesses'], 0)
        self.assertEqual(result['fallback_reasons'], {'none': 1})

    def test_summary_handles_only_fail_open_requests(self):
        rows = [{
            'question_id': 'q0',
            'service_seconds': 12.0,
            'proof_plan': {'evidence_state_prefilter_shadow': {
                'seconds': 0.01,
                'input_candidate_count': 4,
                'filtered_candidate_count': 4,
                'candidate_reduction_fraction': 0.0,
                'frontier_audit': {
                    'baseline_frontier_candidate_count': 2,
                    'filtered_baseline_frontier_candidate_count': None,
                    'baseline_frontier_reduction_fraction': None,
                    'host_witness_count': 0,
                    'host_witness_missing_ids': [],
                    'host_witness_coverage': None,
                    'safe_to_replace_host_grounding': False,
                },
                'exact_metadata': False,
                'applied': False,
                'fallback_reason': 'no_exact_evidence_predicate',
                'answer_path_unchanged': True,
                'coverage_gate_passed': None,
            }},
        }]
        result = summarize(rows)
        self.assertIsNone(result['baseline_frontier_candidate_reduction_fraction'])
        self.assertIsNone(result['applied_host_witness_coverage_rate'])
        self.assertEqual(result['fallback_reasons'], {'no_exact_evidence_predicate': 1})

    def test_summary_separates_proposed_reduction_from_fail_open_effective_reduction(self):
        rows = [{
            'question_id': 'q0',
            'service_seconds': 12.0,
            'proof_plan': {'evidence_state_prefilter_shadow': {
                'seconds': 0.01,
                'input_candidate_count': 4,
                'filtered_candidate_count': 4,
                'candidate_reduction_fraction': 0.0,
                'proposed_candidate_reduction_fraction': 0.5,
                'package_proposed': True,
                'coverage_gate_passed': False,
                'effective_candidate_count': 4,
                'frontier_audit': {
                    'baseline_frontier_candidate_count': 4,
                    'filtered_baseline_frontier_candidate_count': 2,
                    'baseline_frontier_reduction_fraction': 0.5,
                    'host_witness_count': 1,
                    'host_witness_missing_ids': ['host-only'],
                    'host_witness_coverage': False,
                    'safe_to_replace_host_grounding': False,
                },
                'exact_metadata': True,
                'applied': False,
                'fallback_reason': 'host_witness_coverage_failed',
                'answer_path_unchanged': True,
            }},
        }]
        result = summarize(rows)
        self.assertEqual(result['candidate_reduction_fraction']['mean'], 0.0)
        self.assertEqual(result['rows'][0]['proposed_candidate_reduction_fraction'], 0.5)
        self.assertEqual(result['fail_open_after_coverage_failure'], 1)


if __name__ == '__main__':
    unittest.main()
