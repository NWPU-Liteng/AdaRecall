from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_latency_accounting import diagnostic_replay_summary, full_service_summary, measured_e2e_pair


def serving_run():
    return {'complete': True, 'performance_comparable': True, 'verify_generation': False,
        'numerical_probe_overhead_in_service': False, 'evaluation_label_files_read': [],
        'timing_origin': 'measured_server_wall_clock_seconds', 'diagnostic_graph_trace_in_service': False,
        'measurement_contract': {field: 'frozen' for field in ('scope', 'hardware_identity', 'concurrency',
            'startup_policy', 'cohort_registration_sha256', 'input_archive_sha256', 'reader_identity')},
        'rows': [{'question_id': str(index), 'question_sha256': 'q', 'images': ['original'],
            'hypergraph_evidence_plan': {'selected_hyperedges': [index]}, 'answer_token_ids': [2, 3],
            'stop_reason': 'eos', 'service_seconds': 10. + index, 'retrieval_seconds': 2.,
            'prefill_seconds': 3., 'decode_seconds': 5. + index} for index in range(4)]}


class LatencyAccountingTests(unittest.TestCase):
    def test_full_service_includes_all_stages_and_all_requests(self):
        result = full_service_summary([serving_run()])
        self.assertEqual(result['requests'], 4)
        self.assertEqual(result['service_seconds']['sum_seconds'], 46)

    def test_visual_prefill_cannot_be_double_counted(self):
        run = serving_run()
        run['rows'][0]['prefill_seconds'] += 3
        with self.assertRaises(ValueError):
            full_service_summary([run])

    def test_complete_legal_e2e_pair_uses_total_wall_time(self):
        reference = serving_run()
        candidate = deepcopy(reference)
        for row in candidate['rows']:
            row['service_seconds'] /= 2
            row['retrieval_seconds'] /= 2
            row['prefill_seconds'] /= 2
            row['decode_seconds'] /= 2
        self.assertEqual(measured_e2e_pair(reference, candidate)['measured_server_end_to_end_speedup'], 2)

    def test_no_modeled_or_probe_time_may_enter_measured_speedup(self):
        for mutation in ('model', 'trace', 'scope', 'output', 'missing'):
            reference = serving_run()
            candidate = deepcopy(reference)
            if mutation == 'model':
                candidate['timing_origin'] = 'modeled_lpddr_cycles'
            elif mutation == 'trace':
                candidate['diagnostic_graph_trace_in_service'] = True
            elif mutation == 'scope':
                candidate['measurement_contract']['scope'] = 'decode_only'
            elif mutation == 'output':
                candidate['rows'][0]['answer_token_ids'] = [5]
            else:
                candidate['rows'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                measured_e2e_pair(reference, candidate)

    def test_frozen_replay_is_reported_but_not_compared_as_e2e(self):
        timing = {'evidence_retrieval_included': False, 'graph_observation_included': True,
            'performance_comparable': False, 'input_preparation_seconds': 1.,
            'prefill_seconds': 3., 'decode_with_graph_probe_seconds': 4.,
            'frozen_evidence_request_seconds': 8.}
        result = diagnostic_replay_summary([{'rows': [{'diagnostic_latency': timing}]}])
        self.assertIsNone(result['end_to_end_speedup'])

    def test_cache_diagnostics_may_differ_when_semantic_plan_and_output_match(self):
        reference = serving_run()
        candidate = deepcopy(reference)
        for row in candidate['rows']:
            row['hypergraph_evidence_plan']['embedding_store'] = {'hits': 10, 'misses': 1}
        result = measured_e2e_pair(reference, candidate)
        self.assertEqual(result['requests'], 4)

    def test_semantic_text_image_or_hyperedge_identity_changes_fail(self):
        for field in ('text', 'images', 'accepted_atom_ids', 'selected_hyperedges'):
            reference = serving_run()
            candidate = deepcopy(reference)
            candidate['rows'][0]['hypergraph_evidence_plan'][field] = ['changed']
            with self.subTest(field=field), self.assertRaises(ValueError):
                measured_e2e_pair(reference, candidate)

    def test_repeated_question_identity_cannot_inflate_denominator(self):
        reference = serving_run()
        reference['rows'][1]['question_id'] = reference['rows'][0]['question_id']
        with self.assertRaises(ValueError):
            measured_e2e_pair(reference, deepcopy(reference))
