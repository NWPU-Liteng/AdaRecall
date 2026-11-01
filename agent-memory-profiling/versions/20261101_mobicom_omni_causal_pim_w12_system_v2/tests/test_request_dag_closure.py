import sys
from copy import deepcopy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from project_request_dag_closure import project_request_dag


def host_run():
    return {'complete': True, 'evaluation_label_files_read': [], 'rows': [
        {'question_id': 'q0', 'question_sha256': 'q0', 'answer': 'a',
         'answer_token_ids': [1], 'stop_reason': 'eos', 'service_seconds': 100.,
         'ttft_seconds': 90., 'prefill_seconds': 10., 'decode_seconds': 10.},
        {'question_id': 'q1', 'question_sha256': 'q1', 'answer': 'b',
         'answer_token_ids': [2], 'stop_reason': 'eos', 'service_seconds': 80.,
         'ttft_seconds': 70., 'prefill_seconds': 8., 'decode_seconds': 10.},
    ]}


def replay(speedup=2.):
    return {'aggregate': {'linear_phase_speedup_warm': speedup}}


def functional_run():
    run = deepcopy(host_run())
    run['lm_head_functional_dispatch'] = {'physical_pim_executed': False}
    return run


class RequestDagClosureTests(unittest.TestCase):
    def test_favorable_sensitivity_keeps_non_decode_path(self):
        result = project_request_dag(host_run(), replay(1.165), functional_run=functional_run())
        self.assertEqual(result['projection_kind'], 'optimistic_sensitivity_projection')
        self.assertIn('not a physical measurement or strict bound',
                      result['projection_interpretation'])
        self.assertFalse(result['positive_e2e_gate_pass'])
        self.assertEqual(result['gates']['quality_contract_pass'], True)
        self.assertLess(result['speedups']['service'], 1.05)
        self.assertAlmostEqual(result['speedups']['ttft'], 1.165)

    def test_overhead_can_only_reduce_projection(self):
        favorable_case = project_request_dag(host_run(), replay())
        charged = project_request_dag(host_run(), replay(), pim_boundary_seconds=5.)
        self.assertGreater(favorable_case['speedups']['service'], charged['speedups']['service'])
        self.assertEqual(charged['projection_kind'], 'request_level_projection')

    def test_quality_mismatch_blocks_gate(self):
        candidate = functional_run()
        candidate['rows'][1]['answer_token_ids'] = [9]
        result = project_request_dag(host_run(), replay(), functional_run=candidate)
        self.assertFalse(result['gates']['quality_contract_pass'])
        self.assertFalse(result['positive_e2e_gate_pass'])

    def test_invalid_partition_is_rejected(self):
        invalid = host_run()
        invalid['rows'][0]['service_seconds'] = 99.
        with self.assertRaises(ValueError):
            project_request_dag(invalid, replay())


if __name__ == '__main__':
    unittest.main()
