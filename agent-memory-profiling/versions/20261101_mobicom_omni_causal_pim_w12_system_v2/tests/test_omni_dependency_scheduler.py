from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_dependency_scheduler import MaterializationState, solve_exact


class DependencySchedulerTests(unittest.TestCase):
    def test_exact_solution_respects_dependency_closure(self):
        result = solve_exact([
            MaterializationState('image', 4, 5),
            MaterializationState('reader', 6, 12, ('image',)),
            MaterializationState('unrelated', 5, 8),
        ], 10)
        self.assertEqual(result['selected_state_ids'], ['image', 'reader'])
        self.assertEqual(result['net_benefit'], 17)
        self.assertTrue(result['upper_bound_equals_optimum'])

    def test_dependency_cannot_reference_unknown_state(self):
        with self.assertRaises(ValueError):
            solve_exact([MaterializationState('reader', 1, 1, ('missing',))], 4)

    def test_negative_benefit_state_is_not_forced(self):
        result = solve_exact([MaterializationState('bad', 1, -1)], 4)
        self.assertEqual(result['selected_state_ids'], [])
        self.assertEqual(result['net_benefit'], 0)

    def test_cycles_and_unbounded_enumeration_fail_closed(self):
        with self.assertRaises(ValueError):
            solve_exact([MaterializationState('a', 1, 1, ('b',)),
                         MaterializationState('b', 1, 1, ('a',))], 4)
        with self.assertRaises(ValueError):
            solve_exact([MaterializationState(str(index), 1, 1) for index in range(21)], 21)


if __name__ == '__main__':
    unittest.main()
