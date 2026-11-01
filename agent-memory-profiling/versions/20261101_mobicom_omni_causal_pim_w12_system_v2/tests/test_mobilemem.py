import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from causal_materialization import solve_materialization_plan
from mobilemem_prepare import build_record, parse_time
from mobilemem_runtime import dialogue_unions, hybrid_candidates, plan_resident_scan, select_development
from pim_backend import BankPimConfig, BankPimModel, bank_int8_topk


class MobileMemTests(unittest.TestCase):
    def test_observable_only_and_labels_isolated(self):
        record = {'person': {'id': 'p', 'secret': 'FUTURE'}, 'graphs': [{'secret': 'FUTURE'}],
                  'sessions': [{'id': 's', 'messages': [
                      {'id': 'a', 'timestamp': '2024-01-01 00:00:00', 'role': 'user', 'content': 'alpha'},
                      {'id': 'b', 'timestamp': '2024-01-01 00:00:01', 'role': 'assistant', 'content': 'beta'}]}],
                  'question_type_toolbook': {'question_types': [{'qa_pairs': [{
                      'id': 'q', 'effective_timestamp': '2024-01-01 00:00:02', 'question': 'ask',
                      'golden_answers': ['SECRET_ANSWER'], 'question_type': 'single-hop',
                      'source_evidences': [{'id': 'a'}]}]}]}}
        graph, questions, labels = build_record(record)
        self.assertNotIn('SECRET_ANSWER', str(graph) + str(questions))
        self.assertNotIn('FUTURE', str(graph) + str(questions))
        self.assertEqual(graph['dependencies'], {'b': 'a'})
        modified = copy.deepcopy(record)
        modified['question_type_toolbook']['question_types'][0]['qa_pairs'][0]['golden_answers'] = ['CHANGED']
        other_graph, other_questions, _ = build_record(modified)
        self.assertEqual((graph, questions), (other_graph, other_questions))
        self.assertEqual(labels[0]['golden_answers'], ['SECRET_ANSWER'])
        self.assertEqual(parse_time('2024-01-01 00:00:00').utcoffset().total_seconds(), 0)

    def test_materialization_maximizes_closed_bundle_coverage(self):
        result = solve_materialization_plan([(0, 1), (1, 2), (3,)], [2, 1, 1, 4], budget=4)
        self.assertEqual(result['objective'], 2)
        self.assertEqual(result['materialized'], [0, 1, 2])
        self.assertEqual(result['used_tokens'], 4)
        self.assertEqual(result['objective_schema'],
                         'maximum feasible closed-anchor coverage, then minimum shared materialization cost')
        self.assertTrue(result['search_exhausted'])
        self.assertEqual(result['shared_references'], {'0': 1, '1': 2, '2': 1})

    def test_materialization_rejects_infeasible_and_deduplicates(self):
        result = solve_materialization_plan([(0,), (0,), (1,)], [2, 2], budget=2)
        self.assertEqual(result['bundle_count'], 2)
        self.assertEqual(result['objective'], 1)
        rejected = solve_materialization_plan([(0, 1)], [2, 2], budget=1)
        self.assertEqual(rejected['infeasible_closure_indices'], [0])
        self.assertEqual(rejected['materialized'], [])
        self.assertEqual(rejected['used_tokens'], 0)
        self.assertEqual(result['accepted_anchor_ranks'], [0, 1])

    def test_bank_scan_matches_exact_integer_reference(self):
        generator = np.random.default_rng(42)
        for count in (1, 7, 32, 129):
            vectors = generator.integers(-127, 128, size=(count, 64), dtype=np.int8)
            query = generator.integers(-127, 128, size=64, dtype=np.int8)
            indices, scores = bank_int8_topk(vectors, query, banks=32, top_k=16)
            reference = vectors.astype(np.int32) @ query.astype(np.int32)
            wanted = np.lexsort((np.arange(count), -reference))[:16]
            np.testing.assert_array_equal(indices, wanted)
            np.testing.assert_array_equal(scores, reference[wanted])
        model = BankPimModel(BankPimConfig())
        trace = model.resident_score_scan(129, 64, top_k=16, route='pim')
        self.assertEqual(trace['logical_read_bytes'], 129 * 64)
        self.assertGreater(model.copy_cost(129 * 64)['cycles'], 0)

    def test_development_temporal_sampling_and_shards(self):
        questions = [{'question_id': str(index), 'effective_timestamp': f'{index:03d}',
                      'split': 'development' if index < 10 else 'sealed_test'} for index in range(12)]
        selected = select_development(questions, limit=10, sample_count=4)
        self.assertEqual([question['question_id'] for question in selected], ['0', '3', '6', '9'])
        shards = [select_development(questions, limit=10, sample_count=4, shard_count=3, shard_index=index)
                  for index in range(3)]
        self.assertEqual(sum(shards, []), selected)

    def test_pim_residency_separate_from_feature_cache(self):
        model = BankPimModel(BankPimConfig())
        resident = set()
        trace = plan_resident_scan(model, ['alpha', 'beta'], 64, 2, resident)
        self.assertEqual(trace['selected_route'], 'host')
        self.assertEqual(resident, set())
        identities = [str(index) for index in range(512)]
        trace = plan_resident_scan(model, identities, 1024, 128, resident)
        self.assertEqual(trace['selected_route'], 'host')
        self.assertEqual(trace['known_dag_references'], 1)
        self.assertEqual(trace['new_resident_bytes'], 0)
        resident.update(identities)
        trace = plan_resident_scan(model, identities, 1024, 128, resident)
        self.assertEqual(trace['selected_route'], 'pim')
        self.assertEqual(trace['new_resident_bytes'], 0)
        self.assertEqual(trace['cold_copy_cycles'], 0)

    def test_directed_visible_dialogue_prerequisites(self):
        graph = {'dependencies': {'reply': 'ask'}}
        visible = [{'event_id': 'ask', 'identity': 'first'}, {'event_id': 'reply', 'identity': 'second'},
                   {'event_id': 'reply', 'identity': 'third'}]
        self.assertEqual(dialogue_unions(graph, visible, [0, 1]),
                         [frozenset({0}), frozenset({0, 1, 2})])

    def test_candidate_generation_is_dense_only(self):
        atoms = [{'content': 'a', 'event_id': 'a'}, {'content': 'b', 'event_id': 'a'},
                 {'content': 'c', 'event_id': 'c'}]
        selected, meta = hybrid_candidates(atoms, 'ignored', [2, 0, 1], count=2)
        self.assertEqual(selected, [2, 0, 1])
        self.assertEqual(meta['candidate_source'], 'frozen dense event index')
        self.assertEqual(meta['candidate_granularity'], 'event with all visible atom windows')
        self.assertNotIn('lexical_top_indices', meta)


if __name__ == '__main__':
    unittest.main()
