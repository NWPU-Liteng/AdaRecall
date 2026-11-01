import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from omni_proof_backend import choose_backend
from omni_proof_lifecycle import compile_lifecycle
from pim_backend.evidence_state import (
    EvidenceRouteProfile,
    compile_evidence_state,
    plan_evidence_route,
)


def lifecycle_fixture():
    return {
        'schema': 'proof-lifecycle-v1',
        'states': [
            {'state_id': 'text:a', 'kind': 'text', 'resident_bytes_estimate': 100,
             'producer': 'source:a', 'consumers': ['reader_prefill'],
             'release_after': 'reader_prefill', 'recompute': 'not_used',
             'identity_bound': True},
            {'state_id': 'image:b', 'kind': 'image', 'resident_bytes_estimate': 900,
             'producer': 'source:b', 'consumers': ['reader_prefill'],
             'release_after': 'reader_prefill', 'recompute': 'not_used',
             'identity_bound': True},
        ],
        'reader_input_tokens': 64,
    }


class EvidenceStateTests(unittest.TestCase):
    def test_lifecycle_exposes_reader_input_tokens_for_post_generation_contract(self):
        lifecycle = compile_lifecycle(
            {'text:a': {'object_id': 'text:a', 'kind': 'text', 'text': 'hello'}},
            ['text:a'], reader_input_tokens=123, image_bytes={})
        self.assertEqual(lifecycle['reader_input_tokens'], 123)

    def test_compiler_preserves_identity_consumers_and_bytes(self):
        state = compile_evidence_state(lifecycle_fixture(), question_id='q0')
        self.assertEqual(state['schema'], 'evidence-state-frontier-v1')
        self.assertEqual(state['state_count'], 2)
        self.assertEqual(state['unique_source_count'], 2)
        self.assertEqual(state['payload_bytes'], 1000)
        self.assertEqual(state['metadata_bytes'], 64)
        self.assertEqual(state['packed_bytes'], 1064)
        self.assertEqual(state['records'][0]['consumer_mask'], 1)
        self.assertTrue(state['records'][0]['identity_bound'])

    def test_missing_or_unqualified_profile_fails_closed_to_host(self):
        state = compile_evidence_state(lifecycle_fixture())
        without_profile = plan_evidence_route(state)
        self.assertEqual(without_profile['chosen_backend'], 'host')
        self.assertIn('missing_backend_profile', without_profile['reason_codes'])

        profile = EvidenceRouteProfile('uncalibrated', 1000., 10000., 100000,
                                       0., 0., 0., qualified=False, enabled=True)
        route = plan_evidence_route(state, profile=profile, host_stage_seconds=2.,
                                    total_service_seconds=10., quality_contract=True)
        self.assertEqual(route['chosen_backend'], 'host')
        self.assertIn('profile_unqualified', route['reason_codes'])

    def test_negative_net_stage_cost_falls_back_even_when_capacity_fits(self):
        state = compile_evidence_state(lifecycle_fixture())
        profile = EvidenceRouteProfile('slow-pim', 1000., 100., 100000,
                                       0., 0., 0., qualified=True, enabled=True)
        route = plan_evidence_route(state, profile=profile, host_stage_seconds=1.,
                                    total_service_seconds=10., quality_contract=True)
        self.assertEqual(route['chosen_backend'], 'host')
        self.assertIn('not_net_positive', route['reason_codes'])
        self.assertTrue(route['capacity_admitted'])

    def test_all_gates_enable_pim_and_report_end_to_end_bound(self):
        state = compile_evidence_state(lifecycle_fixture(), question_id='q1')
        profile = EvidenceRouteProfile('qualified-pim', 1000., 10000., 100000,
                                       0., 0., 0., qualified=True, enabled=True,
                                       executable=True)
        route = plan_evidence_route(state, profile=profile, host_stage_seconds=2.,
                                    total_service_seconds=10., quality_contract=True)
        self.assertEqual(route['chosen_backend'], 'pim')
        self.assertTrue(route['physical_execution'])
        self.assertEqual(route['route_scope'], 'end_to_end_bound')
        self.assertGreater(route['end_to_end_speedup'], 1.)

    def test_serving_backend_plan_contains_evidence_route_before_reader(self):
        lifecycle = lifecycle_fixture()
        plan = choose_backend(reader_dtype='bf16', object_frontier_bytes=1000,
                              input_tokens=64, lifecycle=lifecycle, question_id='q2')
        self.assertEqual(plan['chosen_backend'], 'host')
        self.assertEqual(plan['phase_routes'][0]['phase'], 'grounding_frontier')
        self.assertEqual(plan['phase_routes'][0]['backend'], 'host')
        self.assertEqual(plan['phase_routes'][1]['phase'], 'evidence_materialize')
        self.assertEqual(plan['phase_routes'][1]['backend'], 'host')
        self.assertEqual(plan['evidence_state']['question_id'], 'q2')
        self.assertIn('missing_backend_profile', plan['evidence_route']['reason_codes'])
        self.assertIn('grounding_frontier_missing', plan['grounding_route']['reason_codes'])


if __name__ == '__main__':
    unittest.main()
