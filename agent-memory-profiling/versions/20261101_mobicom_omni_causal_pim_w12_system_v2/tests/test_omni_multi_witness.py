from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from omni_multi_witness import (CANDIDATE_ONLY, SOURCE_LOCATED,
                                audit_multi_witness, branch_source_families,
                                compile_witness_domains, independent_of_existing,
                                has_independent_branch_pair,
                                limit_witness_alternatives,
                                select_min_cost_branch_cover, source_family)


def demand(identity, variables=None):
    return {'demand_id': identity, 'text': identity,
            'variables': variables or []}


def witness(identity, demand_id, object_ids, *, score=1., bindings=None,
            source_relations=None):
    return {'witness_id': identity, 'demand_id': demand_id,
            'object_ids': object_ids, 'record_ids': [identity],
            'score': score, 'bindings': bindings or {},
            'source_relations': source_relations or []}


def candidate(identity, demand_id, object_ids, *, score=1., bindings=None):
    return {'witness_id': identity, 'demand_id': demand_id,
            'base_record_id': identity, 'object_ids': object_ids,
            'score': score, 'bindings': bindings or {},
            'source_relations': []}


class MultiWitnessTests(unittest.TestCase):
    def test_candidates_remain_separate_and_cannot_upgrade_verification(self):
        domains = compile_witness_domains(
            [demand('d0')], [witness('w0', 'd0', ['o0'])],
            [candidate('c0', 'd0', ['o1']), candidate('c1', 'd0', ['o2'])],
            max_candidate_alternatives=2)
        states = [item['verification_state'] for item in domains['d0']]
        self.assertEqual(states, [SOURCE_LOCATED, CANDIDATE_ONLY, CANDIDATE_ONLY])

    def test_source_located_branch_preserves_typed_values(self):
        item = witness('w0', 'd0', ['o0'])
        item['values'] = [{'name': 'd0.date', 'type': 'date', 'value': '2025-01-01',
                           'record_id': 'w0', 'text_index': 0, 'object_id': 'o0'}]
        item['value_sources'] = list(item['values'])
        domains = compile_witness_domains([demand('d0')], [item])
        self.assertEqual(domains['d0'][0]['values'], item['values'])
        self.assertEqual(domains['d0'][0]['value_sources'], item['value_sources'])

    def test_min_cost_solver_uses_or_alternative(self):
        demands = [demand('d0')]
        domains = {'d0': [
            witness('expensive', 'd0', ['a', 'b'], score=2.),
            witness('cheap', 'd0', ['a'], score=1.),
        ]}
        costs = {'a': 1., 'b': 10.}
        result = select_min_cost_branch_cover(
            demands, domains, [],
            lambda ids: {'feasible': True,
                         'cost': sum(costs[item] for item in ids)})
        self.assertTrue(result['feasible'])
        self.assertEqual([item['witness_id'] for item in result['selected_branches']],
                         ['cheap'])
        self.assertEqual(result['cost'], 1.)

    def test_binding_relation_is_hard_constraint(self):
        demands = [demand('left', ['person']), demand('right', ['person'])]
        domains = {
            'left': [witness('l0', 'left', ['a'], bindings={'person': 'A'})],
            'right': [witness('r0', 'right', ['b'], bindings={'person': 'B'})],
        }
        relation = {'left': 'left', 'right': 'right', 'kind': 'same_binding',
                    'variable': 'person'}
        result = select_min_cost_branch_cover(
            demands, domains, [relation], lambda ids: {'feasible': True,
                                                       'cost': len(ids)})
        self.assertFalse(result['feasible'])

    def test_audit_does_not_call_one_source_an_independent_or(self):
        report = audit_multi_witness(
            [demand('d0')], [witness('w0', 'd0', ['o0'])],
            [candidate('c0', 'd0', ['o1'])], selected_object_ids=['o0'],
            object_costs={'o0': 10., 'o1': 1.})
        self.assertEqual(report['coverage']['demands_with_independent_source_located_or'], 0)
        self.assertEqual(report['coverage']['demands_with_candidate_only_or'], 0)
        self.assertEqual(report['costs']['candidate_only_lower_bound'], 1.)
        self.assertIn('not a quality claim', report['boundary'])

    def test_same_session_atoms_are_not_independent(self):
        first = {'record_ids': ['atom:0_41:turn:1'],
                 'retrieval_record_ids': ['atom:0_41:turn:1']}
        second = {'record_ids': ['atom:0_41:turn:2'],
                  'retrieval_record_ids': ['atom:0_41:turn:2']}
        self.assertEqual(source_family('atom:0_41:turn:1'), 'atom:0_41')
        self.assertEqual(branch_source_families(first), {'atom:0_41'})
        self.assertFalse(has_independent_branch_pair([first, second]))
        self.assertFalse(independent_of_existing(second, [first]))

    def test_disjoint_source_families_form_independent_or(self):
        first = {'record_ids': ['atom:0_41:turn:1'],
                 'retrieval_record_ids': ['atom:0_41:turn:1']}
        second = {'record_ids': ['atom:0_42:turn:1'],
                  'retrieval_record_ids': ['atom:0_42:turn:1']}
        self.assertTrue(has_independent_branch_pair([first, second]))
        self.assertTrue(independent_of_existing(second, [first]))

    def test_same_image_regions_are_not_independent(self):
        first = {'retrieval_record_ids': [
            'visual:' + 'a' * 64 + ':region:2']}
        second = {'retrieval_record_ids': [
            'visual:' + 'a' * 64 + ':region:3']}
        self.assertEqual(source_family(first['retrieval_record_ids'][0]),
                         'image:' + 'a' * 64)
        self.assertFalse(has_independent_branch_pair([first, second]))
        self.assertFalse(independent_of_existing(second, [first]))

    def test_visual_region_and_image_object_share_source_family(self):
        first = {'retrieval_record_ids': [
            'visual:' + 'a' * 64 + ':region:2']}
        second = {'retrieval_record_ids': ['image:' + 'a' * 64]}
        self.assertFalse(has_independent_branch_pair([first, second]))

    def test_different_images_form_independent_or(self):
        first = {'retrieval_record_ids': [
            'visual:' + 'a' * 64 + ':region:2']}
        second = {'retrieval_record_ids': [
            'visual:' + 'b' * 64 + ':region:0']}
        self.assertTrue(has_independent_branch_pair([first, second]))

    def test_witness_alternative_limit_keeps_source_diversity(self):
        values = [
            witness('high-a', 'd0', ['o0'], score=9.),
            witness('high-b', 'd0', ['o1'], score=8.),
            witness('low-a', 'd0', ['o2'], score=1.),
            witness('low-b', 'd0', ['o3'], score=.5),
        ]
        for item, family in zip(values, ('a', 'a', 'b', 'c')):
            item['record_ids'] = ['atom:0_' + family + ':turn:1']
            item['retrieval_record_ids'] = item['record_ids']
        kept = limit_witness_alternatives(values, 3)
        self.assertEqual({item['witness_id'] for item in kept},
                         {'high-a', 'low-a', 'low-b'})

    def test_audit_counts_only_family_disjoint_or(self):
        same_family = [
            witness('w0', 'd0', ['o0']), witness('w1', 'd0', ['o1'])]
        same_family[0]['record_ids'] = ['atom:0_41:turn:1']
        same_family[1]['record_ids'] = ['atom:0_41:turn:2']
        report = audit_multi_witness(
            [demand('d0')], same_family,
            selected_object_ids=['o0', 'o1'],
            object_costs={'o0': 1., 'o1': 1.})
        self.assertEqual(
            report['coverage']['demands_with_independent_source_located_or'], 0)

    def test_audit_counts_family_disjoint_or(self):
        independent = [
            witness('w0', 'd0', ['o0']), witness('w1', 'd0', ['o1'])]
        independent[0]['record_ids'] = ['atom:0_41:turn:1']
        independent[1]['record_ids'] = ['atom:0_42:turn:1']
        report = audit_multi_witness(
            [demand('d0')], independent,
            selected_object_ids=['o0', 'o1'],
            object_costs={'o0': 1., 'o1': 1.})
        self.assertEqual(
            report['coverage']['demands_with_independent_source_located_or'], 1)


if __name__ == '__main__':
    unittest.main()
