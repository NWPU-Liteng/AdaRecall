import itertools
from fractions import Fraction
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from causal_materialization import solve_materialization_plan


class CausalMaterializationTests(unittest.TestCase):
    def test_evidence_objective_matches_exhaustive_rational_search(self):
        generator = random.Random(20261019)
        for _ in range(60):
            costs = [generator.randint(1, 6) for _ in range(7)]
            bundles = [frozenset(generator.sample(range(7), generator.randint(1, 4))) for _ in range(7)]
            scores = [generator.random() for _ in bundles]
            unique = list(dict.fromkeys(bundles))
            ranks = [bundles.index(bundle) for bundle in unique]
            utilities = [max(Fraction(scores[index]) for index, bundle in enumerate(bundles)
                             if bundle == closure) for closure in unique]
            budget = generator.randint(1, 20)
            expected = (Fraction(0), 0, ())
            for mask in itertools.product((False, True), repeat=len(unique)):
                accepted = [index for index, include in enumerate(mask) if include]
                union = set().union(*(unique[index] for index in accepted))
                used = sum(costs[index] for index in union)
                if used <= budget:
                    expected = max(expected, (sum((utilities[index] for index in accepted), Fraction(0)),
                                               -used, tuple(-ranks[index] for index in accepted)))
            result = solve_materialization_plan(bundles, costs, budget=budget, evidence_scores=scores)
            actual = (Fraction(result['objective_units'], result['utility_denominator']),
                      -result['used_tokens'], tuple(-ranks[index] for index in result['accepted_closure_indices']))
            self.assertEqual(actual, expected)
            self.assertTrue(result['optimality_certified'])

    def test_evidence_not_cheap_closure_count_and_no_duplicate_reward(self):
        result = solve_materialization_plan([(0,), (0,), (1,), (2,)], [4, 2, 2], budget=4,
                                            evidence_scores=[0.9, 0.8, 0.1, 0.1])
        self.assertEqual(result['materialized'], [0])
        self.assertEqual(result['accepted_closure_count'], 1)
        self.assertEqual(result['objective'], 0.9)
        self.assertEqual(result['accepted_anchor_ranks'], [0, 1])

    def test_weighted_truncation_bound_and_zero_utility(self):
        result = solve_materialization_plan([(0,), (1,), (2,)], [1, 1, 1], budget=3,
                                            max_nodes=2, evidence_scores=[0.9, 0.4, 0.1])
        self.assertLessEqual(result['used_tokens'], 3)
        self.assertGreaterEqual(Fraction(result['upper_bound_units'], result['utility_denominator']),
                                Fraction(0.9) + Fraction(0.4) + Fraction(0.1))
        self.assertFalse(result['optimality_certified'])
        zero = solve_materialization_plan([(0,)], [1], budget=3, evidence_scores=[0])
        self.assertEqual(zero['materialized'], [])
        for scores in ([], [float('nan')], [-1], [1.1], [float('inf')]):
            with self.assertRaises(ValueError):
                solve_materialization_plan([(0,)], [1], budget=3, evidence_scores=scores)

    def test_search_matches_exhaustive_subsets(self):
        generator = random.Random(20261018)
        for _ in range(60):
            costs = [generator.randint(1, 6) for _ in range(7)]
            bundles = [frozenset(generator.sample(range(7), generator.randint(1, 4))) for _ in range(7)]
            unique = list(dict.fromkeys(bundles))
            ranks = [bundles.index(bundle) for bundle in unique]
            budget = generator.randint(1, 20)
            expected = (0, 0, ())
            for mask in itertools.product((False, True), repeat=len(unique)):
                accepted = [index for index, include in enumerate(mask) if include]
                union = set().union(*(unique[index] for index in accepted))
                used = sum(costs[index] for index in union)
                if used <= budget:
                    expected = max(expected, (len(accepted), -used, tuple(-ranks[index] for index in accepted)))
            result = solve_materialization_plan(bundles, costs, budget=budget)
            actual = (result['objective'], -result['used_tokens'],
                      tuple(-ranks[index] for index in result['accepted_closure_indices']))
            self.assertEqual(actual, expected)
            self.assertTrue(result['optimality_certified'])
            self.assertEqual(result['absolute_gap'], 0)

    def test_truncated_search_has_feasible_incumbent_and_valid_bound(self):
        result = solve_materialization_plan([(0,), (1,), (2,)], [1, 1, 1], budget=3, max_nodes=2)
        self.assertLessEqual(result['used_tokens'], 3)
        self.assertGreaterEqual(result['upper_bound'], 3)
        self.assertFalse(result['optimality_certified'])
        self.assertFalse(result['secondary_objective_certified'])
        self.assertEqual(result['expanded_nodes'], 2)

    def test_invalid_inputs_fail_closed(self):
        for kwargs in ({'max_nodes': 0}, {'max_nodes': 1.5}, {'budget': float('nan')}):
            arguments = {'budget': 3, **kwargs}
            with self.assertRaises(ValueError):
                solve_materialization_plan([(0,)], [1], **arguments)
        for bundle in ([], [1], [-1], [0.5]):
            with self.assertRaises(ValueError):
                solve_materialization_plan([bundle], [1], budget=3)


if __name__ == '__main__':
    unittest.main()
