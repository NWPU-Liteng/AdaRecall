import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from replay_evidence_state_routes import replay_rows, summarize


def row_fixture():
    return {
        'person_id': '0',
        'question_id': '0_q_0',
        'input_tokens': 40,
        'service_seconds': 10.,
        'lifecycle_plan': {
            'schema': 'proof-lifecycle-v1',
            'states': [{'state_id': 'text:a', 'kind': 'text',
                        'resident_bytes_estimate': 100, 'producer': 'source:a',
                        'consumers': ['reader_prefill'], 'release_after': 'reader_prefill',
                        'recompute': 'not_used', 'identity_bound': True}],
        },
        'execution_events': [
            {'phase': 'query_parse_and_retrieval', 'seconds': 1.},
            {'phase': 'source_grounding', 'seconds': 2.},
            {'phase': 'proof_materialize', 'seconds': .5},
        ],
        'proof_plan': {'grounding': {'candidate_count': 6}},
    }


class EvidenceStateReplayTests(unittest.TestCase):
    def test_replay_is_host_only_without_a_qualified_profile(self):
        replayed = replay_rows([row_fixture()])
        self.assertEqual(replayed[0]['route']['chosen_backend'], 'host')
        self.assertEqual(replayed[0]['route']['route_scope'], 'evidence_state_only')
        self.assertAlmostEqual(replayed[0]['state_materialize_seconds'], .5)

    def test_summary_reports_stage_fraction_and_zero_stage_bound(self):
        summary = summarize(replay_rows([row_fixture()]))
        self.assertAlmostEqual(summary['evidence_pipeline_fraction'], .35)
        self.assertAlmostEqual(summary['grounding_fraction'], .2)
        self.assertAlmostEqual(summary['state_materialize_fraction'], .05)
        self.assertAlmostEqual(summary['zero_materialize_speedup_upper_bound'], 10 / 9.5)
        self.assertEqual(summary['route_counts'], {'host': 1})
        self.assertEqual(summary['grounding_candidate_count']['p50'], 6.)

    def test_summary_reports_adaptive_frontier_accounting_when_present(self):
        row = row_fixture()
        row['grounding_frontier'] = {
            'schema': 'evidence-candidate-frontier-v1',
            'input_candidate_count': 6,
            'state_count': 6,
            'packed_bytes': 600,
            'semantic_pruning_applied': False,
        }
        row['grounding_frontier_levels'] = [
            {'level': 0, 'candidate_count': 2, 'packed_bytes': 200},
            {'level': 1, 'candidate_count': 6, 'packed_bytes': 600},
        ]
        row['proof_plan']['grounding']['frontier_selected_level'] = 0
        summary = summarize(replay_rows([row]))
        frontier = summary['candidate_frontier']
        self.assertEqual(frontier['first_level_candidate_count']['p50'], 2.)
        self.assertEqual(frontier['selected_candidate_count']['p50'], 2.)
        self.assertAlmostEqual(frontier['selected_candidate_reduction_fraction']['mean'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
