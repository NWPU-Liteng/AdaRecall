from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_joint_closure import amortization, paired_uncertainty


class ClosureAuditTests(unittest.TestCase):
    def test_negative_warm_saving_has_no_break_even(self):
        result = amortization(10., 20., 1., 2., 100., 2)
        self.assertIsNone(result['linear_extrapolation_break_even_queries'])
        self.assertEqual(result['covered_lifecycle_candidate_seconds'], 122.)

    def test_producer_cost_is_not_hidden_from_lifecycle(self):
        result = amortization(100., 50., 10., 10., 1000., 10)
        self.assertEqual(result['linear_extrapolation_break_even_queries'], 200)
        self.assertGreater(result['covered_lifecycle_candidate_seconds'],
                           result['covered_lifecycle_baseline_seconds'])

    def test_pairing_is_required(self):
        with self.assertRaises(ValueError):
            paired_uncertainty({'a': 'CORRECT'}, {'b': 'CORRECT'})

    def test_proxy_improvement_is_not_population_noninferiority(self):
        result = paired_uncertainty({'a': 'WRONG', 'b': 'CORRECT'},
                                    {'a': 'CORRECT', 'b': 'CORRECT'}, repeats=100)
        self.assertEqual(result['wins'], 1)
        self.assertEqual(result['losses'], 0)
        self.assertFalse(result['population_noninferiority_established'])


if __name__ == '__main__':
    unittest.main()
