import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from pim_backend import BankPimConfig, BankPimModel, plan_materialized_scan
from trajectory_hypergraph import trajectory_unions
from causal_materialization import solve_materialization_plan
from system_audit import materialization_valid


class MaterializationContractTests(unittest.TestCase):
    def test_evidence_score_audit_binds_exact_objective_and_online_scores(self):
        plan = solve_materialization_plan([(0,), (1,)], [4, 2], budget=4,
                                          evidence_scores=[0.9, 0.1])
        plan.update(atom_costs=[4, 2], atom_identities=['first', 'second'])
        row = {'evidence_plan': plan, 'effective_timestamp': '2026-10-19',
               'selected_atoms': [{'identity': 'first', 'timestamp': '2026-10-18'}],
               'retrieval_audit': {'reranker_scores': [0.1, 0.9]}}
        self.assertTrue(materialization_valid(row, 4))
        for field, value in (('objective_units', 0), ('utility_denominator', 0),
                             ('input_evidence_scores', [0.8, 0.1]), ('absolute_gap_units', -1)):
            changed = copy.deepcopy(row)
            changed['evidence_plan'][field] = value
            self.assertFalse(materialization_valid(changed, 4))
        row['retrieval_audit']['reranker_scores'] = [0.2, 0.9]
        self.assertFalse(materialization_valid(row, 4))

    def test_question_text_does_not_change_dependency_closure(self):
        graph = {'dependencies': {'reply': 'ask', 'future': 'ask'}}
        visible = [{'event_id': 'ask', 'identity': 'first'},
                   {'event_id': 'reply', 'identity': 'second'},
                   {'event_id': 'reply', 'identity': 'third'}]
        first = trajectory_unions(graph, visible, [1], 'before after money date')
        second = trajectory_unions(graph, visible, [1], '任意改写问题，不改变观测依赖')
        self.assertEqual(first, second)
        self.assertEqual(first[0], [frozenset({0, 1, 2})])
        self.assertEqual(first[1]['bundles'][0]['bundle_event_ids'], ['ask', 'reply'])

    def test_directed_siblings_share_parent_without_selecting_each_other(self):
        graph = {'dependencies': {'first_reply': 'ask', 'second_reply': 'ask'}}
        visible = [{'event_id': event, 'identity': event} for event in ('ask', 'first_reply', 'second_reply')]
        bundles, _ = trajectory_unions(graph, visible, [1, 2])
        self.assertEqual(bundles, [frozenset({0, 1}), frozenset({0, 2})])
        self.assertEqual(bundles[0] & bundles[1], {0})

    def test_invisible_prerequisite_and_cycle_fail_closed(self):
        visible = [{'event_id': 'reply', 'identity': 'reply'}]
        with self.assertRaises(ValueError):
            trajectory_unions({'dependencies': {'reply': 'missing'}}, visible, [0])
        with self.assertRaises(ValueError):
            trajectory_unions({'dependencies': {'reply': 'reply'}}, visible, [0])

    def test_host_does_not_admit_unpaid_resident_vectors(self):
        backend = BankPimModel(BankPimConfig())
        resident = {'old'}
        before = copy.copy(resident)
        trace = plan_materialized_scan(backend, ['first', 'second'], 64, 2, resident)
        self.assertEqual(trace['selected_route'], 'host')
        self.assertEqual(resident, before)
        self.assertEqual(trace['cold_copy_bytes'], 0)
        self.assertGreater(trace['hypothetical_cold_copy_bytes'], 0)
        self.assertEqual(trace['known_dag_references'], 1)

    def test_capacity_overflow_is_host_fallback(self):
        backend = BankPimModel(BankPimConfig(pim_capacity_bytes=128))
        resident = set()
        trace = plan_materialized_scan(backend, ['first', 'second', 'third'], 64, 3, resident)
        self.assertEqual(trace['selected_route'], 'host')
        self.assertEqual(trace['fallback_reason'], 'capacity_or_workspace')
        self.assertEqual(resident, set())

    def test_duplicate_vector_identity_fails_closed(self):
        backend = BankPimModel(BankPimConfig())
        with self.assertRaises(ValueError):
            plan_materialized_scan(backend, ['first', 'first'], 64, 2, set())


if __name__ == '__main__':
    unittest.main()
