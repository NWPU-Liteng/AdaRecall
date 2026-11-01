import copy
import unittest

from close_evidence_compiler_mainline import evidence_statistics, project_candidate
from pim_backend.lpddr_bank_pim import BankPimConfig


class EvidenceCompilerClosureTests(unittest.TestCase):
    def run_fixture(self, calls=5):
        return {'complete': True, 'reader_quantization': 'bf16',
            'evidence_policy_name': 'evidence_compiler', 'reader_config_sha256': 'reader',
            'reader_weight_bytes': 8388608, 'rows': [{'answer_token_ids': [1, 2, 3]}],
            'operator_trace': {'coverage': 'linear hooks', 'operators': [{
                'name': 'decoder.proj', 'phase': 'decode', 'rows': 1, 'in_features': 2048,
                'out_features': 2048, 'element_bytes': 2, 'weight_bytes': 8388608, 'calls': calls}]}}

    def test_same_trace_aggregate_charges_placement_once(self):
        config = BankPimConfig(pim_capacity_bytes=12 * 1024**2)
        single = project_candidate([self.run_fixture()], config)
        double = project_candidate([self.run_fixture(), self.run_fixture()], config)
        self.assertEqual(double['questions'], 2)
        self.assertEqual(double['decode_tokens_observed'], 6)
        self.assertEqual(double['host_linear_cycles'], single['host_linear_cycles'] * 2)
        self.assertEqual(double['one_time_placement_cycles'], single['one_time_placement_cycles'])
        self.assertEqual(double['cold_hybrid_linear_cycles'],
                         single['warm_hybrid_linear_cycles'] * 2 + single['one_time_placement_cycles'])
        self.assertEqual(double['resident_matrices'], single['resident_matrices'])
        self.assertIsNone(double['end_to_end_speedup'])
        self.assertFalse(double['measured_gpu_seconds_replaced'])

    def test_arithmetic_and_reader_mismatch_are_not_silently_projected(self):
        for key, value in [('reader_quantization', 'w8a8'), ('complete', False),
                           ('reader_config_sha256', 'different'), ('evidence_policy_name', 'local_hyperedge')]:
            run = self.run_fixture()
            run[key] = value
            with self.assertRaises(ValueError):
                project_candidate([self.run_fixture(), run], BankPimConfig())
        run = self.run_fixture()
        run['operator_trace']['operators'][0]['element_bytes'] = 1
        with self.assertRaises(ValueError):
            project_candidate([run], BankPimConfig())

    def test_missing_decode_trace_is_not_imputed(self):
        run = self.run_fixture()
        run['operator_trace']['operators'][0]['phase'] = 'prefill'
        with self.assertRaises(ValueError):
            project_candidate([run], BankPimConfig())

    def test_optional_solver_fields_and_all_images_accounted(self):
        row = {'images': [], 'input_tokens': 100, 'hypergraph_evidence_plan': {}}
        other = copy.deepcopy(row)
        other['images'] = ['image_a', 'image_b']
        other['hypergraph_evidence_plan'] = {'solver_status': 1, 'mip_gap': .2,
                                            'render_feasibility_cuts': [['edge_a']]}
        stats = evidence_statistics([{'rows': [row, other]}])
        self.assertEqual(stats['requests'], 2)
        self.assertEqual(stats['requests_with_images'], 1)
        self.assertEqual(stats['image_inputs'], 2)
        self.assertEqual(stats['solver_status_counts'], {'1': 1})
        self.assertEqual(stats['render_feasibility_cut_count'], 1)
        self.assertEqual(stats['max_reported_mip_gap'], .2)


if __name__ == '__main__':
    unittest.main()
