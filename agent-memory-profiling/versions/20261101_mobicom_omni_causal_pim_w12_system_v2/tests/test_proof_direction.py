import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from audit_proof_direction import call_order, optimistic_stage_bound, repeat_counts, selected_jsonl
from omni_proof_grounding import validate_witnesses
from omni_proof_join import solve_witness_join
from omni_proof_query_contract import (parse_query_contract, reader_context,
                                       render_operation_plan, retrieval_views)
from omni_proof_runtime import authorize_development_users, file_digest


def temporal_contract():
    return {'answer_mode': 'structured', 'source_facts': [
        {'demand_id': 'start', 'text': 'The session start time', 'query_quote': 'start', 'variables': []},
        {'demand_id': 'end', 'text': 'The session end time', 'query_quote': 'end', 'variables': []}],
        'relations': [], 'operations': [
            {'operation_id': 'elapsed', 'kind': 'difference', 'inputs': ['start', 'end'],
             'query_quote': 'how long'}]}


class ProofDirectionTests(unittest.TestCase):
    def test_development_authorization_never_opens_history_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            split = prepared / 'split.json'
            split.write_text(json.dumps({'development_people': ['0', '1', '2'],
                                         'validation_people': ['3'], 'holdout_people': ['6']}))
            (prepared / 'audit.json').write_text(json.dumps({
                'artifacts_sha256': {'split.json': file_digest(split)}}))
            opened = []
            original_open = Path.open

            def guarded_open(path, *arguments, **keywords):
                opened.append(path.name)
                if path.name not in {'audit.json', 'split.json'}:
                    raise AssertionError('unexpected data access: ' + path.name)
                return original_open(path, *arguments, **keywords)

            with patch.object(Path, 'open', guarded_open):
                result = authorize_development_users(prepared, ['0'])
                self.assertIn('artifacts_sha256', result)
                for forbidden in ['3', '6']:
                    with self.assertRaisesRegex(ValueError, 'development users only'):
                        authorize_development_users(prepared, [forbidden])
            self.assertEqual(set(opened), {'audit.json', 'split.json'})

    def test_mainline_does_not_load_the_shared_all_user_history(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/run_proof_mainline.py'
        tree = ast.parse(source.read_text())
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertIn('authorize_development_users', calls)
        self.assertNotIn('authorize_history', calls)

    def test_derived_operation_is_not_a_third_source_requirement(self):
        contract = parse_query_contract(temporal_contract(), 'Between start and end, how long?')
        self.assertEqual(contract['status'], 'ok')
        self.assertEqual(len(contract['source_facts']), 2)
        witnesses = [{'demand_id': identity, 'witness_id': identity, 'object_ids': [identity],
                      'score': 1., 'bindings': {}, 'source_relations': []} for identity in ['start', 'end']]
        result = solve_witness_join(contract['source_facts'], witnesses, [],
                                    lambda identities: {'feasible': True, 'cost': len(identities)})
        self.assertTrue(result['feasible'])
        self.assertEqual(result['accepted_object_ids'], ['end', 'start'])
        self.assertFalse(contract['semantic_entailment_proven'])

    def test_full_question_is_preserved_even_if_fact_description_drops_context(self):
        query = 'During Alice trip on July 2, between start and end, how long?'
        contract = parse_query_contract(temporal_contract(), query)
        for view in retrieval_views(query, contract):
            self.assertEqual(json.loads(view)['original_question'], query)

    def test_operations_cannot_bind_unknown_or_forward_operation(self):
        proposed = temporal_contract()
        proposed['operations'][0]['inputs'] = ['start', 'missing']
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'unknown_operation_input')
        proposed = temporal_contract()
        proposed['operations'].append({'operation_id': 'later', 'kind': 'comparison',
            'inputs': ['elapsed', 'end'], 'query_quote': 'how long'})
        proposed['operations'][0]['inputs'] = ['start', 'later']
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'unknown_operation_input')

    def test_composed_operations_form_a_forward_only_derivation_dag(self):
        query = ('time between start and end; time between second-start and second-end; '
                 'which gap larger')
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'start', 'text': 'start time', 'query_quote': 'start', 'variables': []},
            {'demand_id': 'end', 'text': 'end time', 'query_quote': 'end', 'variables': []},
            {'demand_id': 'start2', 'text': 'second start time', 'query_quote': 'second-start', 'variables': []},
            {'demand_id': 'end2', 'text': 'second end time', 'query_quote': 'second-end', 'variables': []}],
            'relations': [], 'operations': [
                {'operation_id': 'gap0', 'kind': 'difference', 'inputs': ['start', 'end'],
                 'query_quote': 'time between start and end'},
                {'operation_id': 'gap1', 'kind': 'difference', 'inputs': ['start2', 'end2'],
                 'query_quote': 'time between second-start and second-end'},
                {'operation_id': 'choice', 'kind': 'comparison', 'inputs': ['gap0', 'gap1'],
                 'query_quote': 'which gap larger'}]}
        contract = parse_query_contract(proposed, query)
        self.assertEqual(contract['status'], 'ok')
        plan = render_operation_plan(contract)
        self.assertIn('comparison(difference(', plan)
        self.assertIn('difference(start time; end time)', plan)

    def test_full_question_and_operation_plan_reach_reader_context(self):
        query = 'Between start and end, how long?'
        contract = parse_query_contract(temporal_contract(), query)
        context = reader_context(query, '2025-01-01 12:00:00', 'source text',
                                 render_operation_plan(contract))
        self.assertIn(query, context)
        self.assertIn('difference(The session start time; The session end time)', context)

    def test_difference_requires_two_operands(self):
        proposed = temporal_contract()
        proposed['operations'][0]['inputs'] = ['start']
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'missing_operation_operand')

    def test_operation_span_must_be_in_question(self):
        proposed = temporal_contract()
        proposed['operations'][0]['query_quote'] = 'not stated'
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'unlocated_operation')

    def test_source_location_alone_does_not_certify_operation_semantics(self):
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'context', 'text': 'career conversation context',
             'query_quote': 'after career talk', 'variables': []},
            {'demand_id': 'answer', 'text': 'final study-plan direction',
             'query_quote': 'study plan direction', 'variables': []}], 'relations': [],
            'operations': [{'operation_id': 'op0', 'kind': 'comparison',
                'inputs': ['context', 'answer'], 'query_quote': 'study plan direction'}]}
        query = 'after career talk, what was the final study plan direction?'
        contract = parse_query_contract(proposed, query)
        self.assertEqual(contract['status'], 'ok')
        self.assertFalse(contract['semantic_entailment_proven'])

    def test_binary_difference_rejects_three_inputs(self):
        proposed = temporal_contract()
        proposed['source_facts'].append({'demand_id': 'other', 'text': 'other event time',
            'query_quote': 'other', 'variables': []})
        proposed['operations'][0]['inputs'].append('other')
        self.assertEqual(parse_query_contract(proposed, 'start end other how long')['status'],
                         'invalid_difference_arity')

    def test_shared_query_span_does_not_merge_distinct_event_reference_roles(self):
        query = 'For events A and B, which happened later after dinner?'
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'dinner_a', 'text': 'dinner time corresponding to event A',
             'query_quote': 'dinner', 'variables': []},
            {'demand_id': 'dinner_b', 'text': 'dinner time corresponding to event B',
             'query_quote': 'dinner', 'variables': []}], 'relations': [], 'operations': []}
        contract = parse_query_contract(proposed, query)
        self.assertEqual(contract['status'], 'ok')
        self.assertEqual(len(contract['source_facts']), 2)
        proposed['source_facts'][1]['text'] = proposed['source_facts'][0]['text']
        self.assertEqual(parse_query_contract(proposed, query)['status'], 'duplicate_demand_text')

    def test_minimal_contiguous_quote_is_required_for_each_period(self):
        query = '后两天的沟通状态比前两天有什么变化'
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'later', 'text': '后两天沟通状态', 'query_quote': '后两天', 'variables': []},
            {'demand_id': 'earlier', 'text': '前两天沟通状态', 'query_quote': '前两天', 'variables': []}],
            'relations': [], 'operations': [{'operation_id': 'op0', 'kind': 'comparison',
                'inputs': ['later', 'earlier'], 'query_quote': '比前两天有什么变化'}]}
        self.assertEqual(parse_query_contract(proposed, query)['status'], 'ok')

    def test_followup_context_is_not_a_separate_required_observation(self):
        query = 'After discussing career growth with Lin, which direction did my study plan finally take?'
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'd0', 'text': 'direction of final study plan after career discussion with Lin',
             'query_quote': 'which direction did my study plan finally take', 'variables': []}],
            'relations': [], 'operations': []}
        contract = parse_query_contract(proposed, query)
        self.assertEqual(contract['status'], 'ok')
        self.assertEqual(len(contract['source_facts']), 1)

    def test_specific_historical_lists_are_not_forced_into_contextual_mode(self):
        query = 'Which two things mattered most when I checked the parks?'
        proposed = {'answer_mode': 'structured', 'source_facts': [
            {'demand_id': 'd0', 'text': 'the two most important park-selection criteria',
             'query_quote': query, 'variables': []}], 'relations': [], 'operations': []}
        self.assertEqual(parse_query_contract(proposed, query)['status'], 'ok')

    def test_operations_cannot_overwrite_source_identity(self):
        proposed = temporal_contract()
        proposed['operations'][0]['operation_id'] = 'start'
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'duplicate_operation_identity')

    def test_derived_answers_are_not_entity_join_keys(self):
        proposed = temporal_contract()
        proposed['source_facts'][0]['variables'] = ['elapsed']
        self.assertEqual(parse_query_contract(proposed, 'start end how long')['status'],
                         'observation_or_operation_id_used_as_entity_key')

    def test_contextual_queries_do_not_claim_positive_conjunction(self):
        query = 'What would fit my habits?'
        proposed = {'answer_mode': 'contextual', 'source_facts': [
            {'demand_id': 'whole', 'text': query, 'query_quote': query, 'variables': []}],
            'relations': [], 'operations': []}
        self.assertEqual(parse_query_contract(proposed, query)['status'], 'ok')
        self.assertEqual(len(retrieval_views(query, parse_query_contract(proposed, query))), 1)
        proposed['source_facts'][0]['text'] = 'A guessed required preference'
        self.assertEqual(parse_query_contract(proposed, query)['status'],
                         'contextual_query_must_remain_whole')

    def test_invalid_model_plan_stays_explicit_nonproof_context(self):
        contract = parse_query_contract(None, 'Question')
        self.assertEqual(contract['status'], 'invalid_schema')
        self.assertEqual(contract['answer_mode'], 'contextual')
        self.assertFalse(contract['semantic_entailment_proven'])

    def test_repetition_does_not_certify_reader_state_hits(self):
        row = {'materialized_object_ids': ['text:a', 'image:b'], 'images': [{'image_sha256': 'b'}],
               'proof_plan': {'final_render': {'processor_input_sha256': 'input'}}}
        result = repeat_counts([row, deepcopy(row)])
        self.assertEqual(result['repeated_source_object_occurrences'], 2)
        self.assertEqual(result['repeated_raw_image_occurrences'], 1)
        self.assertEqual(result['repeated_final_input_token_hashes'], 1)
        self.assertIsNone(result['qualified_reader_state_hits'])

    def test_stage_envelope_is_arithmetic_not_a_pim_prediction(self):
        self.assertAlmostEqual(optimistic_stage_bound(100., 10.)['zero_time_stage_service_ratio_upper_bound'], 100 / 90)
        with self.assertRaises(ValueError):
            optimistic_stage_bound(10., 10.)

    def test_only_selected_development_jsonl_rows_are_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'questions.jsonl'
            path.write_text('{"question_id":"dev","person_id":"0"}\n'
                            '{"question_id":"sealed","not even valid JSON"}\n')
            self.assertEqual(set(selected_jsonl(path, ['dev'])), {'dev'})
            with self.assertRaises(ValueError):
                selected_jsonl(path, ['missing'])

    def test_execution_order_audit_detects_posthoc_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.py'
            path.write_text('def main():\n    greedy()\n    compile_lifecycle()\n    choose_backend()\n')
            result = call_order(path)
            self.assertTrue(result['lifecycle_compiled_after_generation'])
            self.assertTrue(result['backend_chosen_after_generation'])

    def test_mainline_plans_backend_before_reader_generation(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/run_proof_mainline.py'
        result = call_order(source)
        self.assertLess(result['call_lines']['compile_lifecycle'][0],
                        result['call_lines']['greedy'][0])
        self.assertLess(result['call_lines']['choose_backend'][0],
                        result['call_lines']['greedy'][0])

    def test_exact_source_location_is_not_semantic_entailment(self):
        objects = {'text:unrelated': {'kind': 'text', 'raw_text': 'It was raining.'}}
        record = {'record_id': 'r0', 'object_ids': ['text:unrelated'], 'kind': 'source_text'}
        proposal = {'witnesses': [{'demand_id': 'choice', 'record_ids': ['r0'],
            'anchors': [{'record_id': 'r0', 'object_id': 'text:unrelated', 'quote': 'It was raining.'}],
            'bindings': [], 'source_relations': []}]}
        accepted, rejected = validate_witnesses(proposal,
            [{'demand_id': 'choice', 'text': 'Which date did Alice choose?', 'variables': []}],
            [], {'r0': record}, objects)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]['grounding_status'], 'source_located_not_semantically_verified')

    def test_fixed_synthetic_join_instances_match_exhaustive_objective(self):
        for alternative_count, shared_cost in itertools.product([1, 2, 3], [1, 3, 8]):
            with self.subTest(alternatives=alternative_count, shared_cost=shared_cost):
                demands = [{'demand_id': identity} for identity in ['old', 'new']]
                witnesses = [{'demand_id': identity, 'witness_id': identity + str(index),
                              'object_ids': ['shared', identity + str(index)], 'score': 1. - index / 10,
                              'bindings': {'entity': 'same' if index < 2 else identity}, 'source_relations': []}
                             for identity in ['old', 'new'] for index in range(alternative_count)]
                relation = [{'left': 'old', 'right': 'new', 'kind': 'same_binding', 'variable': 'entity'}]

                def evaluate(identities):
                    cost = sum(shared_cost if identity == 'shared' else 2 for identity in identities)
                    return {'feasible': cost <= shared_cost + 4, 'cost': cost}

                expected = []
                for old in [item for item in witnesses if item['demand_id'] == 'old']:
                    for new in [item for item in witnesses if item['demand_id'] == 'new']:
                        if old['bindings']['entity'] != new['bindings']['entity']:
                            continue
                        identities = set(old['object_ids']) | set(new['object_ids'])
                        assessment = evaluate(identities)
                        if assessment['feasible']:
                            expected.append((min(old['score'], new['score']), -assessment['cost']))
                result = solve_witness_join(demands, witnesses, relation, evaluate)
                self.assertTrue(result['optimal'])
                self.assertEqual((result['support'], -result['cost']), max(expected))


if __name__ == '__main__':
    unittest.main()
