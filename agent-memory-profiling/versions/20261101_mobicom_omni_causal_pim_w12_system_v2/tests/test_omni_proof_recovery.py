import sys
import itertools
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(ROOT))

from omni_proof_grounding import (WITNESS_SCHEMA, WITNESS_VALUES_SCHEMA,
                                  expand_compact_sources, validate_witnesses,
                                  witness_schema)
from omni_proof_join import solve_witness_join
from omni_proof_planner import QUERY_GRAPH_SCHEMA, parse_query_graph, select_retrieval_package
from omni_proof_grounding import WITNESS_SCHEMA
from omni_proof_runtime import parse_object
from omni_proof_sources import build_sources, materialize


class ProofRecoveryTests(unittest.TestCase):
    def test_query_graph_schema_bounds_demand_explosion_and_witness_fanout(self):
        self.assertEqual(QUERY_GRAPH_SCHEMA['properties']['demands']['maxItems'], 4)
        self.assertEqual(WITNESS_SCHEMA['properties']['witnesses']['maxItems'], 24)
        self.assertEqual(WITNESS_SCHEMA['properties']['witnesses']['items']['properties']
                         ['record_ids']['maxItems'], 8)

    def test_query_graph_rejects_duplicate_demand_quotes(self):
        query = 'Which two things changed, and in what order?'
        demands, relations, status = parse_query_graph({'demands': [
            {'demand_id': 'a', 'text': 'first', 'query_quote': 'Which two things', 'variables': []},
            {'demand_id': 'b', 'text': 'second', 'query_quote': 'Which two things', 'variables': []}],
            'relations': []}, query)
        self.assertEqual(status, 'duplicate_demand_quote')
        self.assertEqual(demands[0]['demand_id'], 'soft_query')
        self.assertFalse(relations)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            parse_object('{"witnesses":[],"witnesses":[{}]}')

    def test_query_only_parser_can_express_conjunctive_demands_and_bound_relation(self):
        query = 'Which project changed its deadline, and what was the new date?'
        demands, relations, status = parse_query_graph({
            'demands': [
                {'demand_id': 'a', 'text': 'project with changed deadline',
                 'query_quote': 'Which project changed its deadline', 'variables': ['project']},
                {'demand_id': 'b', 'text': 'new deadline date',
                 'query_quote': 'what was the new date', 'variables': ['project']}],
            'relations': [{'left': 'a', 'right': 'b', 'kind': 'same_binding',
                           'variable': 'project', 'query_quote': 'Which project'}]}, query)
        self.assertEqual(status, 'ok')
        self.assertEqual([item['demand_id'] for item in demands], ['a', 'b'])
        self.assertEqual(relations[0]['variable'], 'project')

    def test_query_graph_rejects_unlocated_relation(self):
        demands, relations, status = parse_query_graph({
            'demands': [{'demand_id': 'a', 'text': 'old', 'query_quote': 'old', 'hard': True, 'variables': []},
                        {'demand_id': 'b', 'text': 'new', 'query_quote': 'new', 'hard': True, 'variables': []}],
            'relations': [{'left': 'a', 'right': 'b', 'kind': 'source_relation',
                           'relation_id': 'r', 'query_quote': 'not in query'}]}, 'old and new')
        self.assertEqual(status, 'unlocated_relation')
        self.assertEqual(len(demands), 1)
        self.assertEqual(relations, [])

    def test_grounding_requires_exact_text_quote(self):
        objects = {'text:a': {'kind': 'text', 'raw_text': 'Alice chose Monday'}}
        record = {'record_id': 'r0', 'kind': 'source_text', 'object_ids': ['text:a']}
        proposal = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'object_id': 'text:a', 'quote': 'Tuesday'}],
            'bindings': [], 'source_relations': []}]}
        accepted, rejected = validate_witnesses(proposal, [{'demand_id': 'd0', 'variables': []}],
            [], {'r0': record}, objects)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'unlocated_claim')

    def test_values_schema_is_opt_in_and_bounded(self):
        self.assertNotIn('values', WITNESS_SCHEMA['properties']['witnesses']
                         ['items']['properties'])
        enabled = witness_schema(True)
        values = enabled['properties']['witnesses']['items']['properties']['values']
        self.assertEqual(values['maxItems'], 8)
        self.assertEqual(values['items']['properties']['type']['enum'], [
            'date', 'datetime', 'number', 'duration', 'count', 'text'])
        self.assertEqual(WITNESS_VALUES_SCHEMA, enabled)
        self.assertNotIn('values', WITNESS_SCHEMA['properties']['witnesses']
                         ['items']['properties'])

    def test_values_must_be_verbatim_and_rejections_are_nonfatal(self):
        text = 'The project started on 2025-01-01.'
        objects = {'text:a': {'kind': 'text', 'raw_text': text}}
        record = {'record_id': 'r0', 'kind': 'source_text', 'object_ids': ['text:a']}
        aliases = {**record, 'record_id': 'r0'}
        proposal = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}],
            'bindings': [],
            'source_relations': [],
            'values': [
                {'name': 'd0.start', 'type': 'date', 'value': '2025-01-01',
                 'record_id': 'r0', 'text_index': 0},
                {'name': 'd0.fake', 'type': 'date', 'value': '2025-01-02',
                 'record_id': 'r0', 'text_index': 0},
            ]}]}
        expanded = expand_compact_sources(proposal, {'r0': aliases}, objects)
        accepted, rejected = validate_witnesses(
            expanded, [{'demand_id': 'd0', 'variables': []}], [], {'r0': aliases}, objects)
        self.assertFalse(rejected)
        self.assertEqual(len(accepted), 1)
        witness = accepted[0]
        self.assertEqual([value['value'] for value in witness['values']], ['2025-01-01'])
        self.assertEqual(len(witness['value_sources']), 1)
        self.assertIn('text:a', witness['object_ids'])
        self.assertIn('unlocated_value', witness['discarded_optional_fields'])

    def test_value_source_object_changes_witness_identity(self):
        objects = {
            'text:a': {'kind': 'text', 'raw_text': 'Met on 2025-01-01.'},
            'text:b': {'kind': 'text', 'raw_text': 'Also on 2025-01-01.'},
        }
        records = {
            'r0': {'record_id': 'r0', 'kind': 'source_text', 'object_ids': ['text:a']},
            'r1': {'record_id': 'r1', 'kind': 'source_text', 'object_ids': ['text:b']},
        }
        first = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}], 'bindings': [],
            'source_relations': [], 'values': [{'name': 'd0.date', 'type': 'date',
                'value': '2025-01-01', 'record_id': 'r0', 'text_index': 0}]}]}
        second = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r1'],
            'anchors': [{'record_id': 'r1', 'text_index': 0}], 'bindings': [],
            'source_relations': [], 'values': [{'name': 'd0.date', 'type': 'date',
                'value': '2025-01-01', 'record_id': 'r1', 'text_index': 0}]}]}
        demands = [{'demand_id': 'd0', 'variables': []}]
        first_witness = validate_witnesses(
            expand_compact_sources(first, records, objects), demands, [], records, objects)[0][0]
        second_witness = validate_witnesses(
            expand_compact_sources(second, records, objects), demands, [], records, objects)[0][0]
        self.assertNotEqual(first_witness['witness_id'], second_witness['witness_id'])

    def test_join_charges_shared_object_once(self):
        demands = [{'demand_id': 'a'}, {'demand_id': 'b'}]
        witnesses = [
            {'witness_id': 'a1', 'demand_id': 'a', 'object_ids': ['shared', 'x'], 'score': .8,
             'bindings': {}, 'source_relations': []},
            {'witness_id': 'b1', 'demand_id': 'b', 'object_ids': ['shared', 'y'], 'score': .7,
             'bindings': {}, 'source_relations': []},
        ]
        result = solve_witness_join(demands, witnesses, [],
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['accepted_object_ids'], ['shared', 'x', 'y'])
        self.assertTrue(result['structure']['includes_global_exact_render_factor'])

    def test_and_demands_join_complementary_sources_but_reject_wrong_binding(self):
        demands = [{'demand_id': 'identity'}, {'demand_id': 'date'}]
        witnesses = [
            {'witness_id': 'identity-a', 'demand_id': 'identity', 'object_ids': ['a'],
             'score': .8, 'bindings': {'project': 'Atlas'}},
            {'witness_id': 'date-wrong', 'demand_id': 'date', 'object_ids': ['b'],
             'score': .99, 'bindings': {'project': 'Borealis'}},
            {'witness_id': 'date-right', 'demand_id': 'date', 'object_ids': ['c'],
             'score': .7, 'bindings': {'project': 'Atlas'}}]
        relations = [{'left': 'identity', 'right': 'date', 'kind': 'same_binding',
                      'variable': 'project'}]
        result = solve_witness_join(demands, witnesses, relations,
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertEqual({item['witness_id'] for item in result['selected_witnesses']},
                         {'identity-a', 'date-right'})
        self.assertEqual(result['accepted_object_ids'], ['a', 'c'])

    def test_or_alternatives_select_one_valid_witness_for_required_demand(self):
        witnesses = [
            {'witness_id': 'preferred', 'demand_id': 'fact', 'object_ids': ['source-a'],
             'score': .9, 'bindings': {}},
            {'witness_id': 'alternative', 'demand_id': 'fact', 'object_ids': ['source-b'],
             'score': .7, 'bindings': {}}]
        result = solve_witness_join([{'demand_id': 'fact'}], witnesses, [],
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertEqual(len(result['selected_witnesses']), 1)
        self.assertEqual(result['selected_witnesses'][0]['witness_id'], 'preferred')

    def test_source_relation_must_be_grounded_by_one_selected_witness(self):
        demands = [{'demand_id': 'before'}, {'demand_id': 'after'}]
        witnesses = [
            {'witness_id': 'before', 'demand_id': 'before', 'object_ids': ['a'],
             'score': 1., 'bindings': {}, 'source_relations': []},
            {'witness_id': 'after-no-link', 'demand_id': 'after', 'object_ids': ['b'],
             'score': .9, 'bindings': {}, 'source_relations': []}]
        relation = [{'left': 'before', 'right': 'after', 'kind': 'source_relation',
                     'relation_id': 'updated_to'}]
        result = solve_witness_join(demands, witnesses, relation,
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertFalse(result['feasible'])

    def test_nonproof_package_can_include_multiple_sources_with_exact_budget(self):
        records = [
            {'record_id': 'first', 'object_ids': ['a'], 'score': .9},
            {'record_id': 'second', 'object_ids': ['b'], 'score': .8},
            {'record_id': 'third', 'object_ids': ['c'], 'score': .7}]
        package = select_retrieval_package(records,
            lambda identities: {'feasible': len(identities) <= 2, 'cost': len(identities)})
        self.assertEqual(package['selected_record_ids'], ['first', 'second'])
        self.assertFalse(package['semantic_proof'])
        self.assertEqual(package['accepted_object_ids'], ['a', 'b'])

    def test_invalid_query_graph_is_explicit_and_not_a_one_record_proof(self):
        demands, relations, status = parse_query_graph(None, 'What changed?')
        self.assertEqual(status, 'invalid_parse')
        self.assertFalse(relations)
        self.assertTrue(demands[0]['required'])
        package = select_retrieval_package([], lambda identities: {
            'feasible': True, 'cost': 0, 'exact_input_tokens': 25, 'text': '', 'text_tokens': 0,
            'visual_tokens': 0})
        self.assertFalse(package['semantic_proof'])
        self.assertEqual(package['selected_records'], [])

    def test_soft_witness_can_have_multiple_complementary_sources(self):
        objects = {'text:a': {'kind': 'text', 'raw_text': 'First event'},
                   'text:b': {'kind': 'text', 'raw_text': 'Second event'}}
        aliases = {alias: {'record_id': alias, 'kind': 'source_text', 'object_ids': [identity]}
                   for alias, identity in [('r0', 'text:a'), ('r1', 'text:b')]}
        proposal = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0', 'r1'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}, {'record_id': 'r1', 'text_index': 0}],
            'bindings': [{'variable': 'irrelevant', 'value': 'invented', 'record_id': 'r0', 'text_index': 0}],
            'source_relations': []}]}
        accepted, rejected = validate_witnesses(expand_compact_sources(proposal, aliases, objects),
            [{'demand_id': 'd0', 'required': True, 'variables': []}], [], aliases, objects)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]['object_ids'], ['text:a', 'text:b'])
        self.assertEqual(accepted[0]['discarded_optional_fields'], ['unlocated_binding'])

    def test_one_source_can_support_two_demands_without_duplicate_cost(self):
        demands = [{'demand_id': 'name'}, {'demand_id': 'date'}]
        witnesses = [{'witness_id': demand['demand_id'], 'demand_id': demand['demand_id'],
            'object_ids': ['same-atom'], 'score': 1., 'bindings': {}} for demand in demands]
        result = solve_witness_join(demands, witnesses, [],
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertEqual(result['accepted_object_ids'], ['same-atom'])
        self.assertEqual(result['cost'], 1)
        self.assertEqual(len(result['selected_witnesses']), 2)

    def test_only_explicitly_anchored_objects_are_materialized(self):
        objects = {'text:a': {'kind': 'text', 'raw_text': 'Atlas deadline Monday'},
                   'text:b': {'kind': 'text', 'raw_text': 'Unrelated neighbor'}}
        aliases = {'r0': {'record_id': 'r0', 'kind': 'source_text',
            'object_ids': ['text:a'], 'retrieval_context_object_ids': ['text:b']}}
        proposal = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}], 'bindings': [], 'source_relations': []}]}
        accepted, rejected = validate_witnesses(expand_compact_sources(proposal, aliases, objects),
            [{'demand_id': 'd0', 'required': True, 'variables': []}], [], aliases, objects)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]['object_ids'], ['text:a'])

    def test_selected_image_locator_materializes_its_raw_image(self):
        objects = {
            'text:a': {'kind': 'text', 'raw_text': 'The living-room photo is relevant'},
            'image:sha': {'kind': 'image'},
        }
        aliases = {'r0': {
            'record_id': 'visual:sha:region:2',
            'kind': 'image_locator',
            'object_ids': ['text:a'],
            'image_object_id': 'image:sha',
            'locator_regions': [{'region_index': 0}],
            'image_source': {'member': 'image/uid4/event/4_event_8.png'},
        }}
        proposal = {'witnesses': [{
            'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}],
            'bindings': [], 'source_relations': [],
        }]}
        expanded = expand_compact_sources(proposal, aliases, objects)
        accepted, rejected = validate_witnesses(
            expanded, [{'demand_id': 'd0', 'required': True, 'variables': []}],
            [], aliases, objects)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]['object_ids'], ['image:sha', 'text:a'])
        self.assertEqual(accepted[0]['image_locator_status'], 'original_image_required')
        self.assertEqual(
            accepted[0]['source_hints']['image:sha']['member'],
            'image/uid4/event/4_event_8.png')

    def test_duplicate_image_regions_charge_one_raw_image(self):
        witnesses = [
            {'witness_id': 'region-0', 'demand_id': 'first', 'object_ids': ['image:sha'],
             'score': 1., 'bindings': {}},
            {'witness_id': 'region-1', 'demand_id': 'second', 'object_ids': ['image:sha'],
             'score': 1., 'bindings': {}}]
        result = solve_witness_join([{'demand_id': 'first'}, {'demand_id': 'second'}], witnesses, [],
            lambda identities: {'feasible': len(identities) <= 1, 'cost': 512 * len(identities)}, time_limit=1.)
        self.assertEqual(result['accepted_object_ids'], ['image:sha'])
        self.assertEqual(result['cost'], 512)

    def test_conjunction_cannot_silently_drop_a_required_source_for_budget(self):
        witnesses = [{'witness_id': identity, 'demand_id': identity, 'object_ids': [identity],
                      'score': 1., 'bindings': {}} for identity in ('first', 'second')]
        result = solve_witness_join([{'demand_id': 'first'}, {'demand_id': 'second'}], witnesses, [],
            lambda identities: {'feasible': len(identities) <= 1, 'cost': len(identities)}, time_limit=1.)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['selected_witnesses'], [])

    def test_malformed_model_identifiers_are_rejected_not_hashed(self):
        accepted, rejected = validate_witnesses({'witnesses': [{'demand_id': []}]}, [], [], {}, {})
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'unknown_demand')

    def test_text_candidates_are_not_whole_session_containers(self):
        atoms = [{'atom_id': 'turn-' + str(index), 'ordinal': index, 'session_id': 's', 'event_id': 'e',
                  'role': 'user', 'content_type': 'text', 'content': 'Fact ' + str(index), 'image_refs': []}
                 for index in range(6)]
        source = build_sources({'person_id': '0', 'atoms': atoms}, {})
        self.assertEqual(len(source['records']), len(atoms))
        self.assertTrue(all(len(record['object_ids']) == 1 for record in source['records']))
        self.assertTrue(all(len(record['retrieval_context_object_ids']) <= 2 for record in source['records']))

    def test_compact_anchors_expand_to_original_source_not_model_text(self):
        objects = {'text:a': {'kind': 'text', 'raw_text': 'Alice chose Monday'}}
        aliases = {'r0': {'record_id': 'r0', 'kind': 'source_text', 'object_ids': ['text:a']}}
        proposal = {'witnesses': [{'demand_id': 'd0', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'text_index': 0}],
            'bindings': [], 'source_relations': []}]}
        expanded = expand_compact_sources(proposal, aliases, objects)
        self.assertEqual(expanded['witnesses'][0]['anchors'][0]['quote'], 'Alice chose Monday')
        demands = [{'demand_id': 'd0', 'hard': False, 'variables': ['unbound_soft_phrase']}]
        accepted, rejected = validate_witnesses(expanded, demands, [], aliases, objects)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        demands[0]['hard'] = True
        accepted, rejected = validate_witnesses(expanded, demands, [], aliases, objects)
        self.assertEqual(len(accepted), 1)
        relations = [{'kind': 'same_binding', 'left': 'd0', 'right': 'd1', 'variable': 'unbound_soft_phrase'}]
        accepted, rejected = validate_witnesses(expanded, demands, relations, aliases, objects)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'missing_required_binding')

    def test_wrong_project_bindings_do_not_join(self):
        demands = [{'demand_id': 'old'}, {'demand_id': 'new'}]
        witnesses = [
            {'witness_id': 'old', 'demand_id': 'old', 'object_ids': ['a'], 'score': 1.,
             'bindings': {'project': 'A'}},
            {'witness_id': 'wrong', 'demand_id': 'new', 'object_ids': ['b'], 'score': 1.,
             'bindings': {'project': 'B'}},
            {'witness_id': 'right', 'demand_id': 'new', 'object_ids': ['c'], 'score': .5,
             'bindings': {'project': 'A'}}]
        result = solve_witness_join(demands, witnesses, [],
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertEqual(result['support'], .5)
        self.assertEqual(result['accepted_object_ids'], ['a', 'c'])

    def test_source_relation_is_not_invented_by_join(self):
        demands = [{'demand_id': 'old'}, {'demand_id': 'new'}]
        witnesses = [{'witness_id': identity, 'demand_id': identity,
                      'object_ids': [identity], 'score': 1., 'bindings': {}, 'source_relations': []}
                     for identity in ('old', 'new')]
        relations = [{'left': 'old', 'right': 'new', 'kind': 'source_relation', 'relation_id': 'update'}]
        result = solve_witness_join(demands, witnesses, relations,
            lambda identities: {'feasible': True, 'cost': len(identities)}, time_limit=1.)
        self.assertFalse(result['feasible'])

    def test_join_matches_small_cartesian_oracle_for_nonadditive_cost(self):
        demands = [{'demand_id': identity} for identity in ('a', 'b', 'c')]
        witnesses = []
        for demand in demands:
            for index, score in enumerate((.4, .9, .9)):
                witnesses.append({'witness_id': demand['demand_id'] + str(index),
                    'demand_id': demand['demand_id'], 'score': score, 'bindings': {},
                    'object_ids': ['shared', demand['demand_id']] if index == 2 else
                                  [demand['demand_id'] + str(index)]})
        def evaluate(identities):
            cost = len(identities) + (3 if {'a0', 'b0'} <= identities else 0)
            return {'feasible': cost <= 6, 'cost': cost}
        oracle = []
        domains = [[witness for witness in witnesses if witness['demand_id'] == demand['demand_id']]
                   for demand in demands]
        for choice in itertools.product(*domains):
            identities = frozenset(identity for witness in choice for identity in witness['object_ids'])
            assessment = evaluate(identities)
            if assessment['feasible']:
                oracle.append((min(witness['score'] for witness in choice), -assessment['cost']))
        result = solve_witness_join(demands, witnesses, [], evaluate, time_limit=1.)
        self.assertTrue(result['optimal'])
        self.assertEqual((result['support'], -result['cost']), max(oracle))

    def test_five_image_container_does_not_force_sibling_images(self):
        atoms = [{'atom_id': 'text', 'ordinal': 0, 'session_id': 's', 'event_id': 'e',
                  'role': 'user', 'content_type': 'text', 'content': 'Five pages', 'image_refs': []}]
        for index in range(5):
            atoms.append({'atom_id': 'page' + str(index), 'ordinal': index + 1,
                'session_id': 's', 'event_id': 'e', 'role': 'user', 'content_type': 'image',
                'content': '', 'image_refs': [{'status': 'exact', 'normalized': str(index) + '.png'}]})
        graph = build_sources({'person_id': '0', 'atoms': atoms},
                              {str(index) + '.png': 'hash' + str(index) for index in range(5)})
        selected = next(record for record in graph['records'] if record.get('image_object_id') == 'image:hash4')
        delivered = materialize(graph['objects'], selected['object_ids'])
        self.assertEqual(len(graph['containers'][0]['atom_ids']), 6)
        self.assertEqual(len(delivered['images']), 1)
        self.assertEqual(delivered['images'][0]['source_atom_id'], 'page4')


if __name__ == '__main__':
    unittest.main()
