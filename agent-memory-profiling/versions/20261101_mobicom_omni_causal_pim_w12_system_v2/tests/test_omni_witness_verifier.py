from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from omni_multi_witness import CANDIDATE_ONLY, SEMANTIC_VERIFIED
from omni_witness_verifier import (DIRECT_SUPPORT, VERIFIER_JSON_SCHEMA,
                                   audit_verified_frontier, demand_operation,
                                   validate_verifier_result, verified_branch_record,
                                   verifier_payload)


def demand():
    return {'demand_id': 'd0', 'text': '用户喜欢在周末安静地露营',
            'query_quote': '喜欢什么', 'variables': []}


def branch(record_id='atom:0_1:turn:1', family='atom:0_1'):
    return {'demand_id': 'd0', 'record_ids': [record_id],
            'retrieval_record_ids': [record_id], 'source_family': family,
            'verification_state': CANDIDATE_ONLY, 'semantic_sufficiency': 'unverified'}


def text_source():
    return {'record_id': 'atom:0_1:turn:1', 'role': 'user',
            'session_interval': '2025-01-01..2025-01-02',
            'text': '我喜欢周末找安静的营地露营，避开人多的景点。'}


class WitnessVerifierTests(unittest.TestCase):
    def test_payload_contains_one_demand_and_one_branch(self):
        payload = verifier_payload(demand(), branch(), [text_source()])
        self.assertEqual(payload['demand']['demand_id'], 'd0')
        self.assertEqual(payload['branch']['source_family'], ['atom:0_1'])
        self.assertNotIn('question', payload)
        self.assertEqual(VERIFIER_JSON_SCHEMA['additionalProperties'], False)

    def test_commonality_demand_carries_closed_operation(self):
        value = {'demand_id': 'd0', 'text': '婚礼筹备中让用户踏实的共同点',
                 'query_quote': '共同点', 'variables': []}
        self.assertEqual(demand_operation(value), 'commonality')
        payload = verifier_payload(value, branch(), [text_source()])
        self.assertEqual(payload['demand']['operation'], 'commonality')
        self.assertIn('repeated_instances', payload['demand']['operation_rule'])

    def test_temporal_compound_demand_is_not_treated_as_single_fact(self):
        compound = {'demand_id': 'd0', 'text': '1月29日傍晚到站后开始和我爸商量接站',
                    'query_quote': '1月29日傍晚到站后', 'variables': []}
        self.assertEqual(demand_operation(compound), 'compound_fact')
        later_recommendation = {
            'demand_id': 'd0', 'text': '后来有没有再推荐过别的批发渠道',
            'query_quote': '再推荐过别的批发渠道', 'variables': []}
        self.assertEqual(demand_operation(later_recommendation), 'single_fact')

    def test_compound_fact_requires_two_distinct_event_records(self):
        compound = {'demand_id': 'd0', 'text': '1月29日傍晚到站后开始和我爸商量接站',
                    'query_quote': '1月29日傍晚到站后', 'variables': []}
        compound_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_9:turn:0', 'atom:0_9:turn:1'],
            'retrieval_record_ids': ['atom:0_9:turn:0', 'atom:0_9:turn:1'],
        }
        texts = {
            'atom:0_9:turn:0': '我刚到老家县城车站，外面人特别多。',
            'atom:0_9:turn:1': '我一边拖行李一边通过微信和父亲沟通接站和拼车方案。',
        }
        partial = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_9:turn:0',
                          'quote': '我刚到老家县城车站'}]},
            compound_branch, texts, demand=compound)
        self.assertFalse(partial['accepted'])
        self.assertEqual(partial['error'], 'focus_gate_incomplete_compound')
        complete = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [
                {'record_id': 'atom:0_9:turn:0', 'quote': '我刚到老家县城车站'},
                {'record_id': 'atom:0_9:turn:1', 'quote': '沟通接站和拼车方案'},
            ]},
            compound_branch, texts, demand=compound)
        self.assertTrue(complete['accepted'])

    def test_exact_quote_is_accepted(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1',
                          'quote': '找安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        self.assertTrue(result['accepted'])
        self.assertIsNone(result['error'])

    def test_paraphrased_quote_is_rejected(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1',
                          'quote': '用户喜欢安静露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'fabricated_or_nonliteral_quote')

    def test_evidence_outside_branch_is_rejected(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_2:turn:1', 'quote': '找安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text'],
                       'atom:0_2:turn:1': text_source()['text']})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'evidence_not_in_branch')

    def test_sufficient_requires_evidence(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT, 'evidence': []},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'sufficient_without_evidence')

    def test_image_required_remains_unverified(self):
        result = validate_verifier_result({
            'verdict': 'abstain', 'reason_code': 'image_required', 'evidence': []},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        self.assertFalse(result['accepted'])
        promoted = verified_branch_record(branch(), result)
        self.assertEqual(promoted['verification_state'], CANDIDATE_ONLY)
        self.assertEqual(promoted['semantic_sufficiency'], 'unverified')

    def test_single_episode_cannot_prove_commonality(self):
        commonality = {'demand_id': 'd0', 'text': '婚礼筹备中让用户踏实的共同点',
                       'query_quote': '共同点', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=commonality)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_missing_generalization')

    def test_explicit_generalization_can_pass_commonality_gate(self):
        commonality = {'demand_id': 'd0', 'text': '用户每次露营的共同点',
                       'query_quote': '共同点', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1',
                          'quote': '我每次露营都喜欢找安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': '我每次露营都喜欢找安静的营地露营。'},
            demand=commonality)
        self.assertTrue(result['accepted'])

    def test_repeated_instances_from_two_families_can_pass_commonality_gate(self):
        commonality = {'demand_id': 'd0', 'text': '用户多次露营的共同点',
                       'query_quote': '共同点', 'variables': []}
        repeated_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_41_4:turn:43', 'atom:0_39:turn:10'],
            'retrieval_record_ids': ['atom:0_41_4:turn:43', 'atom:0_39:turn:10'],
        }
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'repeated_instances', 'shared_property': '同一顶旧帐篷',
            'evidence': [
                {'record_id': 'atom:0_41_4:turn:43',
                 'quote': '我露营时都会带上同一顶旧帐篷'},
                {'record_id': 'atom:0_39:turn:10',
                 'quote': '每次露营我都带着同一顶旧帐篷'},
            ]},
            repeated_branch, {
                'atom:0_41_4:turn:43': '我露营时都会带上同一顶旧帐篷。',
                'atom:0_39:turn:10': '每次露营我都带着同一顶旧帐篷。',
            }, demand=commonality)
        self.assertTrue(result['accepted'])

    def test_repeated_instances_from_one_family_are_rejected(self):
        commonality = {'demand_id': 'd0', 'text': '用户多次露营的共同点',
                       'query_quote': '共同点', 'variables': []}
        repeated_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_9:turn:1', 'atom:0_9:turn:2'],
            'retrieval_record_ids': ['atom:0_9:turn:1', 'atom:0_9:turn:2'],
        }
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'repeated_instances', 'shared_property': '同一顶旧帐篷',
            'evidence': [
                {'record_id': 'atom:0_9:turn:1', 'quote': '带着同一顶旧帐篷'},
                {'record_id': 'atom:0_9:turn:2', 'quote': '带着同一顶旧帐篷'},
            ]},
            repeated_branch, {
                'atom:0_9:turn:1': '这次露营带着同一顶旧帐篷。',
                'atom:0_9:turn:2': '后来又带着同一顶旧帐篷。',
            }, demand=commonality)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_insufficient_repetition')

    def test_repeated_instances_require_shared_property_in_every_quote(self):
        commonality = {'demand_id': 'd0', 'text': '用户多次露营的共同点',
                       'query_quote': '共同点', 'variables': []}
        repeated_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_41_4:turn:43', 'atom:0_39:turn:10'],
            'retrieval_record_ids': ['atom:0_41_4:turn:43', 'atom:0_39:turn:10'],
        }
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'repeated_instances', 'shared_property': '同一顶旧帐篷',
            'evidence': [
                {'record_id': 'atom:0_41_4:turn:43', 'quote': '带着同一顶旧帐篷'},
                {'record_id': 'atom:0_39:turn:10', 'quote': '去了山里的营地'},
            ]},
            repeated_branch, {
                'atom:0_41_4:turn:43': '这次露营带着同一顶旧帐篷。',
                'atom:0_39:turn:10': '后来又去了山里的营地。',
            }, demand=commonality)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_insufficient_repetition')

    def test_repeated_instances_require_two_records(self):
        commonality = {'demand_id': 'd0', 'text': '用户多次露营的共同点',
                       'query_quote': '共同点', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'repeated_instances', 'shared_property': '同一顶旧帐篷',
            'evidence': [
                {'record_id': 'atom:0_41_4:turn:43', 'quote': '带着同一顶旧帐篷'},
            ]},
            branch('atom:0_41_4:turn:43', 'atom:0_41_4'),
            {'atom:0_41_4:turn:43': '这次露营带着同一顶旧帐篷。'}, demand=commonality)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_insufficient_repetition')

    def test_causal_outcome_without_cause_is_rejected(self):
        reason = {'demand_id': 'd0', 'text': '为什么用户喜欢安静露营',
                  'query_quote': '为什么', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=reason)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_missing_cause')

    def test_reason_accepts_a_role_typed_cause_outcome_pair(self):
        reason = {'demand_id': 'd0', 'text': '生日那天中午为什么没有出去吃饭',
                  'query_quote': '为什么', 'variables': []}
        pair_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_51:turn:0', 'atom:0_51:turn:27'],
            'retrieval_record_ids': ['atom:0_51:turn:0', 'atom:0_51:turn:27'],
        }
        texts = {
            'atom:0_51:turn:0': '我今天中午没出去吃，想在家随便点点外卖。',
            'atom:0_51:turn:27': '对，我就是想简单一点，晚上再和家里人吃顿饭，就够了。',
        }
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'cause_outcome_pair',
            'evidence': [
                {'record_id': 'atom:0_51:turn:0', 'quote': '我今天中午没出去吃',
                 'evidence_role': 'outcome'},
                {'record_id': 'atom:0_51:turn:27', 'quote': '晚上再和家里人吃顿饭',
                 'evidence_role': 'cause'},
            ]},
            pair_branch, texts, demand=reason)
        self.assertTrue(result['accepted'])

    def test_reason_pair_requires_two_records(self):
        reason = {'demand_id': 'd0', 'text': '生日那天中午为什么没有出去吃饭',
                  'query_quote': '为什么', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'cause_outcome_pair',
            'evidence': [
                {'record_id': 'atom:0_51:turn:0', 'quote': '我今天中午没出去吃',
                 'evidence_role': 'outcome'},
                {'record_id': 'atom:0_51:turn:0', 'quote': '晚上再和家里人吃顿饭',
                 'evidence_role': 'cause'},
            ]},
            branch('atom:0_51:turn:0', 'atom:0_51'),
            {'atom:0_51:turn:0': '我今天中午没出去吃，晚上再和家里人吃顿饭。'},
            demand=reason)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_missing_cause_outcome_roles')

    def test_reason_pair_requires_a_causal_or_plan_cue(self):
        reason = {'demand_id': 'd0', 'text': '生日那天中午为什么没有出去吃饭',
                  'query_quote': '为什么', 'variables': []}
        pair_branch = {
            'demand_id': 'd0',
            'record_ids': ['atom:0_51:turn:0', 'atom:0_51:turn:27'],
            'retrieval_record_ids': ['atom:0_51:turn:0', 'atom:0_51:turn:27'],
        }
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'cause_outcome_pair',
            'evidence': [
                {'record_id': 'atom:0_51:turn:0', 'quote': '我今天中午没出去吃',
                 'evidence_role': 'outcome'},
                {'record_id': 'atom:0_51:turn:27', 'quote': '今天天气不错',
                 'evidence_role': 'cause'},
            ]},
            pair_branch, {
                'atom:0_51:turn:0': '我今天中午没出去吃，想在家随便点点外卖。',
                'atom:0_51:turn:27': '今天天气不错，适合散步。',
            }, demand=reason)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_missing_cause')

    def test_support_mode_must_match_the_demand_operation(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'support_mode': 'cause_outcome_pair',
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营',
                          'evidence_role': 'outcome'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'support_mode_operation_mismatch')

    def test_text_cannot_certify_an_image_demand(self):
        image_demand = {'demand_id': 'd0', 'text': '截图里的预算上限',
                        'query_quote': '预算上限', 'variables': [],
                        'evidence_scope': {'kind': 'image_locator'}}
        self.assertEqual(demand_operation(image_demand), 'image_required')
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=image_demand)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_image_requires_visual')

    def test_explicit_date_must_match_the_source_session(self):
        dated = {'demand_id': 'd0', 'text': '9月25日结算体检费时想到的压力',
                 'query_quote': '9月25日', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=dated,
            temporal_context={'session_interval': '2025-09-23 11:00:00..2025-09-23 15:30:00'})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_date_mismatch')

    def test_stale_state_cannot_certify_latest_state(self):
        latest = {'demand_id': 'd0', 'text': '用户当前最新的家庭分工想法',
                  'query_quote': '最新想法', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=latest,
            temporal_context={'session_end': '2025-06-10 21:00:00',
                              'latest_session_end': '2025-09-28 22:00:00'})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['error'], 'focus_gate_not_latest_state')

    def test_latest_session_can_satisfy_latest_state_gate(self):
        latest = {'demand_id': 'd0', 'text': '用户当前最新的家庭分工想法',
                  'query_quote': '最新想法', 'variables': []}
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']}, demand=latest,
            temporal_context={'session_end': '2025-09-28 22:00:00',
                              'latest_session_end': '2025-09-28 22:00:00'})
        self.assertTrue(result['accepted'])

    def test_accepted_branch_is_explicitly_promoted(self):
        result = validate_verifier_result({
            'verdict': 'sufficient', 'reason_code': DIRECT_SUPPORT,
            'evidence': [{'record_id': 'atom:0_1:turn:1', 'quote': '安静的营地露营'}]},
            branch(), {'atom:0_1:turn:1': text_source()['text']})
        promoted = verified_branch_record(branch(), result)
        self.assertEqual(promoted['verification_state'], SEMANTIC_VERIFIED)
        self.assertEqual(promoted['semantic_sufficiency'], 'text_quote_verified')

    def test_two_verified_families_form_an_or(self):
        rows = {
            'd0': [
                {'accepted': True, 'source_family': 'atom:0_1', 'reason_code': DIRECT_SUPPORT},
                {'accepted': True, 'source_family': 'atom:0_2', 'reason_code': DIRECT_SUPPORT},
            ],
        }
        audit = audit_verified_frontier(rows)
        self.assertEqual(audit['demands_with_semantically_verified_or'], 1)
        self.assertEqual(audit['accepted_branch_count'], 2)

    def test_same_family_does_not_form_an_or(self):
        rows = {
            'd0': [
                {'accepted': True, 'source_family': 'atom:0_1', 'reason_code': DIRECT_SUPPORT},
                {'accepted': True, 'source_family': 'atom:0_1', 'reason_code': DIRECT_SUPPORT},
            ],
        }
        audit = audit_verified_frontier(rows)
        self.assertEqual(audit['demands_with_semantically_verified_or'], 0)

    def test_rejected_claim_cannot_form_an_or(self):
        rows = {
            'd0': [
                {'accepted': True, 'source_family': 'atom:0_1', 'reason_code': DIRECT_SUPPORT},
                {'accepted': False, 'source_family': 'atom:0_2',
                 'reason_code': 'partial_support'},
            ],
        }
        audit = audit_verified_frontier(rows)
        self.assertEqual(audit['demands_with_semantically_verified_or'], 0)

    def test_same_demand_id_across_questions_is_not_pooled(self):
        rows = {
            ('q1', 'd0'): [
                {'question_id': 'q1', 'demand_id': 'd0', 'accepted': True,
                 'source_family': 'atom:0_1', 'reason_code': DIRECT_SUPPORT},
            ],
            ('q2', 'd0'): [
                {'question_id': 'q2', 'demand_id': 'd0', 'accepted': True,
                 'source_family': 'atom:0_2', 'reason_code': DIRECT_SUPPORT},
            ],
        }
        audit = audit_verified_frontier(rows)
        self.assertEqual(audit['demand_count'], 2)
        self.assertEqual(audit['demands_with_semantically_verified_or'], 0)
        self.assertEqual({item['question_id'] for item in audit['per_demand']}, {'q1', 'q2'})


if __name__ == '__main__':
    unittest.main()
