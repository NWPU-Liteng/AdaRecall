import itertools
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from omni_proof_join_noisy_or import (calibrated_probability,
                                      solve_configured_witness_join,
                                      solve_noisy_or_saturate,
                                      witness_effective_probability)


DEMANDS = [
    {'demand_id': 'd0'},
    {'demand_id': 'd1'},
    {'demand_id': 'd2'},
]
RELATIONS = [{
    'left': 'd0', 'right': 'd1', 'kind': 'same_binding', 'variable': 'x',
}]


def witness(identity, demand_id, object_ids, probability, binding):
    return {
        'witness_id': identity,
        'demand_id': demand_id,
        'object_ids': object_ids,
        'bindings': {'x': binding},
    }


WITNESSES = [
    witness('w0', 'd0', ['a', 'b'], 0.9, 'A'),
    witness('w1', 'd0', ['c'], 0.8, 'A'),
    witness('w2', 'd1', ['d'], 0.7, 'A'),
    witness('w3', 'd1', ['e'], 0.6, 'B'),
    witness('w4', 'd2', ['f'], 0.5, 'A'),
    witness('w5', 'd2', ['a', 'g'], 0.9, 'A'),
]
PROBABILITIES = {
    item['witness_id']: item_probability
    for item, item_probability in zip(
        WITNESSES, [0.9, 0.8, 0.7, 0.6, 0.5, 0.9])
}


def compatible(left, right):
    overlap = set(left['bindings']) & set(right['bindings'])
    if any(left['bindings'][key] != right['bindings'][key] for key in overlap):
        return False
    return True


def compatible_with_all(candidate, selected):
    return all(compatible(candidate, previous) for previous in selected)


def coverage(selected):
    values = {}
    for demand in DEMANDS:
        demand_id = demand['demand_id']
        remaining = 1.0
        for item in selected:
            if item['demand_id'] == demand_id:
                remaining *= 1.0 - PROBABILITIES[item['witness_id']]
        values[demand_id] = 1.0 - remaining
    return values


class NoisyOrJoinTests(unittest.TestCase):
    def evaluate(self, object_ids, budget=5):
        cost = len(object_ids)
        return {'feasible': cost <= budget, 'cost': cost}

    def test_solver_is_feasible_compatible_and_budgeted(self):
        plan = solve_noisy_or_saturate(
            DEMANDS, WITNESSES, RELATIONS, self.evaluate,
            probability=lambda item: PROBABILITIES[item['witness_id']],
            levels=20, max_witnesses_per_demand=2, time_limit=2.0)

        self.assertTrue(plan['feasible'])
        self.assertLessEqual(plan['cost'], 5)
        selected = plan['selected_witnesses']
        for left, right in itertools.combinations(selected, 2):
            self.assertTrue(compatible(left, right))
        self.assertTrue(all(
            plan['per_demand_coverage'][demand['demand_id']] >= plan['t_star'] - 1e-12
            for demand in DEMANDS))
        self.assertEqual(
            plan['n_evaluate_calls'], plan['evaluated_unique_object_unions'])

        best = 0.0
        domains = [[item for item in WITNESSES if item['demand_id'] == demand['demand_id']]
                   for demand in DEMANDS]
        for counts in itertools.product(range(3), repeat=len(DEMANDS)):
            if any(count > 2 for count in counts):
                continue
            choices = [list(itertools.combinations(domain, count))
                       for domain, count in zip(domains, counts)]
            for subset in itertools.product(*choices):
                selected = [item for group in subset for item in group]
                object_ids = {object_id for item in selected
                              for object_id in item['object_ids']}
                if len(object_ids) > 5:
                    continue
                if any(not compatible(left, right)
                       for left, right in itertools.combinations(selected, 2)):
                    continue
                values = coverage(selected)
                best = max(best, min(values.values()))
        ratio = min(plan['per_demand_coverage'].values()) / best
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0 + 1e-12)

    def test_solver_is_deterministic(self):
        def run():
            plan = solve_noisy_or_saturate(
                DEMANDS, WITNESSES, RELATIONS, self.evaluate,
                probability=lambda item: PROBABILITIES[item['witness_id']],
                levels=13, max_witnesses_per_demand=2, time_limit=2.0)
            return {
                'ids': [item['witness_id'] for item in plan['selected_witnesses']],
                't_star': plan['t_star'],
                'coverage': plan['per_demand_coverage'],
                'cost': plan['cost'],
                'calls': plan['n_evaluate_calls'],
            }
        self.assertEqual(run(), run())

    def test_fill_redundancy_does_not_reduce_coverage(self):
        common = dict(
            probability=lambda item: PROBABILITIES[item['witness_id']],
            levels=20, max_witnesses_per_demand=2, time_limit=2.0)
        without = solve_noisy_or_saturate(
            DEMANDS, WITNESSES, RELATIONS, self.evaluate,
            fill_redundancy=False, **common)
        with_fill = solve_noisy_or_saturate(
            DEMANDS, WITNESSES, RELATIONS, self.evaluate,
            fill_redundancy=True, **common)
        self.assertTrue(with_fill['feasible'])
        self.assertLessEqual(with_fill['cost'], 5)
        self.assertGreaterEqual(
            sum(with_fill['per_demand_coverage'].values()),
            sum(without['per_demand_coverage'].values()) - 1e-12)

    def test_dfs_dispatch_calls_the_original_v1_solver(self):
        called = {}

        def v1(demands, witnesses, relations, evaluate, **kwargs):
            called['kwargs'] = kwargs
            return {'solver': 'v1'}

        result = solve_configured_witness_join(
            'dfs_maxmin', DEMANDS, WITNESSES, RELATIONS, self.evaluate,
            v1_solver=v1, time_limit=1.5, max_states=17)
        self.assertEqual(result, {'solver': 'v1'})
        self.assertEqual(called['kwargs'], {'time_limit': 1.5, 'max_states': 17})

    def test_witness_probability_is_min_over_calibrated_records(self):
        calibrator = {
            'calibration': {
                'text': {'breakpoints': {
                    'rerank_score': [0.0, 1.0],
                    'calibrated_probability': [0.0, 1.0],
                }},
                'image': {'breakpoints': {
                    'rerank_score': [0.0, 1.0],
                    'calibrated_probability': [0.0, 1.0],
                }},
            }
        }
        self.assertEqual(calibrated_probability(calibrator, 'text', 0.25), 0.25)
        item = {'record_scores': [
            {'modality': 'text', 'rerank_score': 0.9},
            {'modality': 'image', 'rerank_score': 0.2},
        ]}
        self.assertEqual(
            witness_effective_probability(item, calibrator), 0.2)


if __name__ == '__main__':
    unittest.main()
