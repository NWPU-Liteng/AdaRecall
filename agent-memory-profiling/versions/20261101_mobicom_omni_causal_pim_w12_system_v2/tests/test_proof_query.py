import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.proof_query import (audit_host_grounding_coverage,
                                     compile_candidate_table, compile_proof_query,
                                     execute_proof_query)


def candidate(identity, demand, *, score, bindings=None, source_relations=None):
    return {
        'witness_id': identity,
        'base_record_id': identity + ':base',
        'demand_id': demand,
        'object_ids': ['text:' + identity],
        'bindings': bindings or {},
        'source_relations': source_relations or [],
        'score': score,
    }


class ProofQueryTests(unittest.TestCase):
    def test_compile_is_deterministic_and_does_not_claim_semantics(self):
        plan = compile_proof_query(
            demands=[{'demand_id': 'd0', 'text': 'first fact', 'variables': ['event']}],
            answer_mode='structured')
        self.assertEqual(plan['schema'], 'proof-query-plan-v1')
        self.assertEqual(len(plan['plan_sha256']), 64)
        self.assertFalse(plan['semantic_claims'])
        self.assertEqual(plan['execution_contract'], 'source_identity_and_constraint_filter_only')

    def test_source_relation_filter_has_a_conservative_empty_frontier(self):
        plan = compile_proof_query(
            demands=[
                {'demand_id': 'd0', 'text': 'first fact', 'variables': []},
                {'demand_id': 'd1', 'text': 'second fact', 'variables': []},
            ],
            relations=[{'relation_id': 'rel0', 'kind': 'source_relation',
                        'left': 'd0', 'right': 'd1'}])
        table = compile_candidate_table([
            candidate('a', 'd0', score=1.0), candidate('b', 'd1', score=0.9)])
        result = execute_proof_query(plan, table)
        self.assertEqual(result['eligible_candidate_count'], 0)
        self.assertEqual(result['missing_demands'], ['d0', 'd1'])
        self.assertEqual(result['unsatisfied_constraints'], ['rel0'])
        self.assertFalse(result['safe_to_replace_host_grounding'])

    def test_same_binding_join_preserves_only_compatible_tuples(self):
        plan = compile_proof_query(
            demands=[
                {'demand_id': 'd0', 'text': 'first', 'variables': ['project']},
                {'demand_id': 'd1', 'text': 'second', 'variables': ['project']},
            ],
            relations=[{'relation_id': 'rel0', 'kind': 'same_binding',
                        'left': 'd0', 'right': 'd1', 'variable': 'project'}])
        table = compile_candidate_table([
            candidate('a', 'd0', score=1.0, bindings={'project': 'p1'}),
            candidate('b', 'd1', score=0.9, bindings={'project': 'p1'}),
            candidate('c', 'd1', score=0.8, bindings={'project': 'p2'}),
        ])
        result = execute_proof_query(plan, table)
        self.assertEqual(result['proof_tuple_count'], 1)
        self.assertEqual(result['eligible_base_record_ids'], ['a:base', 'b:base'])

    def test_host_coverage_requires_identity_preservation(self):
        plan = compile_proof_query(
            demands=[{'demand_id': 'd0', 'text': 'first', 'variables': []}])
        table = compile_candidate_table([candidate('a', 'd0', score=1.0)])
        result = execute_proof_query(plan, table)
        covered = audit_host_grounding_coverage(result, [
            {'retrieval_record_ids': ['a:base']}])
        missing = audit_host_grounding_coverage(result, [
            {'retrieval_record_ids': ['not-present']}])
        self.assertTrue(covered['coverage'])
        self.assertFalse(covered['safe_to_replace_host_grounding'])
        self.assertEqual(covered['reason'], 'host_witness_inside_structural_frontier_but_semantics_unproven')
        self.assertFalse(missing['coverage'])
        self.assertEqual(missing['reason'], 'host_witness_outside_frontier')


if __name__ == '__main__':
    unittest.main()
