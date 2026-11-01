import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.evidence_prefilter import (audit_evidence_prefilter,
                                            enforce_host_witness_coverage,
                                            plan_evidence_state_prefilter)


def candidate(identity, demand, *, binding, score, container=None, session=None):
    return {
        'witness_id': identity,
        'base_record_id': identity,
        'record_id': identity,
        'demand_id': demand,
        'object_ids': ['text:' + identity],
        'bindings': {'project': binding},
        'source_relations': [],
        'score': score,
        'kind': 'source_text',
        'session_id': session or 'session:' + identity,
        'container_id': container or 'exchange:' + identity,
        'primary_object_id': 'text:' + identity,
        'source_atom_ids': ['atom:' + identity],
        'source_span': [0, 1],
    }


class EvidencePrefilterTests(unittest.TestCase):
    def test_no_exact_relation_fails_open(self):
        candidates = [candidate('a', 'd0', binding='p1', score=1.0)]
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'fact', 'variables': ['project']}],
            relations=[], operations=[], candidates=candidates)
        self.assertFalse(result['applied'])
        self.assertEqual(result['fallback_reason'], 'no_exact_evidence_predicate')
        self.assertEqual(result['candidates'], candidates)

    def test_same_binding_filters_full_candidate_pool(self):
        candidates = [
            candidate('a', 'd0', binding='p1', score=1.0),
            candidate('b', 'd1', binding='p1', score=0.9),
            candidate('c', 'd1', binding='p2', score=0.8),
        ]
        result = plan_evidence_state_prefilter(
            demands=[
                {'demand_id': 'd0', 'text': 'first', 'variables': ['project']},
                {'demand_id': 'd1', 'text': 'second', 'variables': ['project']},
            ],
            relations=[{'relation_id': 'rel0', 'kind': 'same_binding',
                        'left': 'd0', 'right': 'd1', 'variable': 'project'}],
            operations=[], candidates=candidates)
        self.assertTrue(result['applied'])
        self.assertEqual(result['filtered_candidate_count'], 2)
        self.assertEqual([item['base_record_id'] for item in result['candidates']], ['a', 'b'])
        self.assertTrue(result['result']['structural_contract_complete'])

    def test_incomplete_exact_metadata_fails_open(self):
        candidates = [
            candidate('a', 'd0', binding='p1', score=1.0),
            {**candidate('b', 'd1', binding='p1', score=0.9), 'bindings': {}},
        ]
        result = plan_evidence_state_prefilter(
            demands=[
                {'demand_id': 'd0', 'text': 'first', 'variables': ['project']},
                {'demand_id': 'd1', 'text': 'second', 'variables': ['project']},
            ],
            relations=[{'relation_id': 'rel0', 'kind': 'same_binding',
                        'left': 'd0', 'right': 'd1', 'variable': 'project'}],
            operations=[], candidates=candidates)
        self.assertFalse(result['applied'])
        self.assertEqual(result['fallback_reason'], 'binding_metadata_incomplete')
        self.assertEqual(result['candidates'], candidates)

    def test_explicit_image_scope_filters_exact_source_kind(self):
        candidates = [
            {**candidate('a', 'd0', binding='p1', score=1.0), 'kind': 'source_text'},
            {**candidate('b', 'd0', binding='p1', score=0.9), 'kind': 'image_locator'},
        ]
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=candidates)
        self.assertTrue(result['applied'])
        self.assertEqual(result['filtered_candidate_count'], 1)
        self.assertEqual(result['candidates'][0]['base_record_id'], 'b')

    def test_explicit_scope_expands_exact_cross_source_session_package(self):
        candidates = [
            {**candidate('image', 'd0', binding='p1', score=1.0,
                         container='exchange:image', session='session:shared'), 'kind': 'image_locator',
             'primary_object_id': 'image:shared', 'object_ids': ['image:shared']},
            candidate('caption', 'd0', binding='p1', score=0.9,
                      container='exchange:text', session='session:shared'),
            candidate('unrelated', 'd0', binding='p1', score=0.8),
        ]
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=candidates)
        self.assertTrue(result['applied'])
        self.assertEqual(result['seed_candidate_count'], 1)
        self.assertEqual(result['filtered_candidate_count'], 2)
        self.assertEqual([item['base_record_id'] for item in result['candidates']],
                         ['image', 'caption'])

    def test_frontier_audit_reports_package_reduction_and_host_witness_coverage(self):
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=[
                {**candidate('text-a', 'd0', binding='p1', score=1.0),
                 'kind': 'source_text'},
                {**candidate('image-a', 'd0', binding='p1', score=0.9),
                 'kind': 'image_locator'},
            ])
        audit = audit_evidence_prefilter(
            result,
            baseline_candidates=[
                candidate('text-a', 'd0', binding='p1', score=1.0),
                candidate('image-a', 'd0', binding='p1', score=0.9),
            ],
            host_witnesses=[{'retrieval_record_ids': ['image-a']}])
        self.assertEqual(audit['baseline_frontier_candidate_count'], 2)
        self.assertEqual(audit['filtered_baseline_frontier_candidate_count'], 1)
        self.assertEqual(audit['baseline_frontier_reduction_fraction'], 0.5)
        self.assertTrue(audit['host_witness_coverage'])
        self.assertFalse(audit['safe_to_replace_host_grounding'])
        enforced = enforce_host_witness_coverage(
            result, audit, [
                candidate('text-a', 'd0', binding='p1', score=1.0),
                candidate('image-a', 'd0', binding='p1', score=0.9),
            ])
        self.assertTrue(enforced['coverage_gate_passed'])
        self.assertFalse(enforced['fallback_to_baseline'])
        self.assertEqual(enforced['effective_frontier_reduction_fraction'], 0.5)

    def test_failed_host_coverage_forces_fail_open(self):
        candidates = [
            {**candidate('text-a', 'd0', binding='p1', score=1.0),
             'kind': 'source_text'},
            {**candidate('image-a', 'd0', binding='p1', score=0.9),
             'kind': 'image_locator'},
        ]
        planned = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=candidates)
        audit = audit_evidence_prefilter(
            planned, candidates, [{'retrieval_record_ids': ['text-a']}])
        enforced = enforce_host_witness_coverage(planned, audit, candidates)
        self.assertFalse(enforced['applied'])
        self.assertEqual(enforced['fallback_reason'], 'host_witness_coverage_failed')
        self.assertEqual(enforced['effective_candidate_count'], len(candidates))
        self.assertTrue(enforced['fallback_to_baseline'])
        self.assertEqual(enforced['frontier_audit']['host_witness_missing_ids'], ['text-a'])

    def test_frontier_audit_flags_host_witness_loss(self):
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=[
                {**candidate('text-a', 'd0', binding='p1', score=1.0),
                 'kind': 'source_text'},
                {**candidate('image-a', 'd0', binding='p1', score=0.9),
                 'kind': 'image_locator'},
            ])
        audit = audit_evidence_prefilter(
            result, baseline_candidates=[candidate('text-a', 'd0', binding='p1', score=1.0)],
            host_witnesses=[{'retrieval_record_ids': ['text-a']}])
        self.assertFalse(audit['host_witness_coverage'])
        self.assertEqual(audit['host_witness_missing_ids'], ['text-a'])

    def test_frontier_audit_marks_fallback_coverage_as_not_applicable(self):
        result = {'applied': False, 'result': None}
        audit = audit_evidence_prefilter(
            result, baseline_candidates=[candidate('a', 'd0', binding='p1', score=1.0)],
            host_witnesses=[])
        self.assertIsNone(audit['baseline_frontier_reduction_fraction'])
        self.assertIsNone(audit['host_witness_coverage'])

    def test_empty_host_witness_set_does_not_pass_coverage_gate(self):
        result = plan_evidence_state_prefilter(
            demands=[{'demand_id': 'd0', 'text': 'photo fact', 'variables': [],
                      'evidence_scope': {'kind': 'image_locator'}}],
            relations=[], operations=[], candidates=[
                {**candidate('text-a', 'd0', binding='p1', score=1.0),
                 'kind': 'source_text'},
                {**candidate('image-a', 'd0', binding='p1', score=0.9),
                 'kind': 'image_locator'},
            ])
        audit = audit_evidence_prefilter(result, result['candidates'], [])
        enforced = enforce_host_witness_coverage(result, audit, result['candidates'])
        self.assertIsNone(enforced['frontier_audit']['host_witness_coverage'])
        self.assertFalse(enforced['applied'])
        self.assertEqual(enforced['fallback_reason'], 'no_host_witnesses_to_validate')
        self.assertEqual(enforced['candidate_reduction_fraction'], 0.0)


if __name__ == '__main__':
    unittest.main()
