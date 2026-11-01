from copy import deepcopy
from pathlib import Path
import sys
import json
import hashlib
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_service_pair import screening_summary, validate_deferred_finite_validation


def stream():
    row = dict(question_id='q', question_sha256='question', images=[], answer_token_ids=[1],
               stop_reason='max_new_tokens', input_tokens=10, visual_patches=0,
               answer_sha256='answer', hypergraph_evidence_plan={}, service_seconds=10.,
               edge_embedding_seconds=1., decode_seconds=3., query_retrieval_seconds=.5,
               evidence_planning_seconds=.5, prefill_seconds=5.,
               service_stage_ledger={'disjoint_seconds': {'evidence_seconds': 2.,
                   'input_preparation_seconds': 0., 'prefill_seconds': 5., 'decode_seconds': 3.},
                   'nested_seconds': {'vision_in_prefill_seconds': 4.}})
    return {'0': dict(complete=True, evaluation_label_files_read=[], rows=[row],
                      construction_seconds=1., model_load_seconds=1., embedding_state_reuse={})}


class ScreeningPairTests(unittest.TestCase):
    def test_deferred_finite_checks_match_decode_ledger(self):
        run = stream()['0']
        run['rows'][0]['answer_token_ids'] = [1, 2]
        run.update(reader_quantization='w8a8', reader_integer_primary={'modules': 2},
                   operator_trace={'operators': [dict(phase='decode', name=name, rows=1,
                                                      element_bytes=1, calls=1)
                                                 for name in ('first', 'last')]},
                   reader_finite_validation={
                       'decode_forward_calls': 1,
                       'finite_predicates_checked': 4,
                       'host_confirmations': 1,
                       'maximum_predicate_payload_bytes': 4,
                       'arithmetic_changed': False,
                       'checks_removed': False,
                       'boundary': 'contract'})
        result = validate_deferred_finite_validation(run)
        self.assertEqual(result['finite_predicates_checked'], 4)

    def test_deferred_finite_checks_reject_mismatched_aggregate(self):
        run = stream()['0']
        run['rows'][0]['answer_token_ids'] = [1, 2]
        run.update(reader_quantization='w8a8', reader_integer_primary={'modules': 2},
                   operator_trace={'operators': [dict(phase='decode', name=name, rows=1,
                                                      element_bytes=1, calls=1)
                                                 for name in ('first', 'last')]},
                   reader_finite_validation={
                       'decode_forward_calls': 1,
                       'finite_predicates_checked': 3,
                       'host_confirmations': 1,
                       'maximum_predicate_payload_bytes': 4,
                       'arithmetic_changed': False,
                       'checks_removed': False,
                       'boundary': 'contract'})
        with self.assertRaises(ValueError):
            validate_deferred_finite_validation(run)

    def test_deferred_finite_checks_reject_missing_or_noninteger_operator_trace(self):
        run = stream()['0']
        run['rows'][0]['answer_token_ids'] = [1, 2]
        run.update(reader_quantization='w8a8', reader_integer_primary={'modules': 1},
                   reader_finite_validation={
                       'decode_forward_calls': 1, 'finite_predicates_checked': 2,
                       'host_confirmations': 1, 'maximum_predicate_payload_bytes': 2,
                       'arithmetic_changed': False, 'checks_removed': False})
        with self.assertRaisesRegex(ValueError, 'operator trace'):
            validate_deferred_finite_validation(run)
        run['operator_trace'] = {'operators': [dict(phase='decode', name='first', rows=1,
                                                   element_bytes=2, calls=1)]}
        with self.assertRaisesRegex(ValueError, 'operator trace'):
            validate_deferred_finite_validation(run)

    def test_functional_backend_call_ledger_and_no_emulator_wallclock_speedup(self):
        reference, candidate = stream(), stream()
        run = candidate['0']
        run.update(reader_functional_backend='pimnast_bank_layout_reference',
                   reader_quantization='w8a8', physical_pim_execution=False,
                   performance_comparable=False,
                   pim_functional_startup={'projection_path': '/frozen/projection.json',
                       'projection_sha256': 'projection', 'declared_weight_pool_bytes': 1024,
                       'layout_resident_bytes': 512, 'functional_extra_cuda_bytes': 128,
                       'layout_seconds_server': 4.})
        run['pim_functional_dispatch'] = {'selected_layouts': 1,
            'resident_layout_bytes': 512, 'calls_per_module': {'linear': 0},
            'prefill_or_multirow_host_fallback_calls': 1,
            'functional_max_workspace_bound_bytes': 64,
            'unobserved_selected_modules': ['linear']}
        run['rows'][0]['person_id'] = '0'
        run['rows'][0]['pim_functional_calls'] = {'linear': 0}
        projection = {'complete': True, 'operand_bits': 8, 'per_user': [{
            'run': 'frozen/reference_user_0.json', 'declared_pim_pool_bytes': 1024,
            'group_residency_policy': {'future_call_count_used_by_objective': False},
            'matrices': [{'name': 'linear', 'shape': '16x16', 'model_route': 'pim',
                          'candidate_resident_bytes': 512}]}]}
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory) / 'projection.json'
            route.write_text(json.dumps(projection))
            run['pim_functional_startup']['projection_path'] = str(route)
            run['pim_functional_startup']['projection_sha256'] = hashlib.sha256(route.read_bytes()).hexdigest()
            result = screening_summary(reference, candidate, {'0': ['q']})
            self.assertIsNone(result['screening_service_ratio'])
            summary = result['per_user'][0]['candidate_functional_backend']
            self.assertEqual(summary['bank_layout_replacement_calls'], 0)
            self.assertFalse(summary['full_device_capacity_qualified'])
            for mutate in (
                lambda value: value['rows'][0]['pim_functional_calls'].update(linear=1),
                lambda value: value['pim_functional_dispatch']['calls_per_module'].update(linear=1),
                lambda value: value.update(physical_pim_execution=True),
                lambda value: value.update(performance_comparable=True),
                lambda value: value['pim_functional_startup'].update(layout_resident_bytes=1)):
                changed = deepcopy(candidate)
                mutate(changed['0'])
                with self.assertRaises(ValueError):
                    screening_summary(reference, changed, {'0': ['q']})

    def test_bounds_keep_truncations_and_do_not_certify_acceleration(self):
        result = screening_summary(stream(), stream(), {'0': ['q']})
        self.assertAlmostEqual(result['reference'][
            'zero_cost_remove_entire_embedding_stage_speedup_upper_bound'], 10 / 9)
        self.assertEqual(result['reference']['truncated_question_ids'], ['q'])
        self.assertEqual(result['reference']['ttft_boundary_seconds']['sum_seconds'], 7.)
        self.assertIsNone(result['measured_causal_end_to_end_speedup'])

    def test_changed_generation_is_reported_not_silently_discarded(self):
        candidate = stream()
        candidate['0']['rows'][0]['answer_token_ids'] = [2]
        result = screening_summary(stream(), candidate, {'0': ['q']})
        self.assertEqual(result['requests_per_policy'], 1)
        self.assertIn('answer_token_ids', result['pair_mismatches'][0]['fields'])

    def test_actual_token_boundaries_are_not_inferred_for_old_runs(self):
        reference, candidate = stream(), stream()
        candidate['0']['rows'][0].update(ttft_seconds=7.1, ttlt_seconds=9.9)
        result = screening_summary(reference, candidate, {'0': ['q']})
        self.assertIsNone(result['reference']['ttft_seconds'])
        self.assertEqual(result['reference']['token_boundary_missing_question_ids'], ['q'])
        self.assertEqual(result['candidate']['ttft_seconds']['mean_seconds'], 7.1)
        self.assertEqual(result['candidate']['ttlt_seconds']['mean_seconds'], 9.9)

    def test_token_boundaries_must_be_paired_finite_and_ordered(self):
        for first, last in ((7., None), (None, 9.), (9., 7.), (7., 11.), (float('nan'), 9.)):
            candidate = stream()
            candidate['0']['rows'][0].update(ttft_seconds=first, ttlt_seconds=last)
            with self.assertRaises(ValueError):
                screening_summary(stream(), candidate, {'0': ['q']})

    def test_missing_and_repeated_requests_or_invalid_ledger_fail(self):
        for expected in ([], ['q', 'q']):
            with self.assertRaises(ValueError):
                screening_summary(stream(), stream(), {'0': expected})
        candidate = deepcopy(stream())
        candidate['0']['rows'][0]['service_seconds'] = 11.
        with self.assertRaises(ValueError):
            screening_summary(stream(), candidate, {'0': ['q']})
