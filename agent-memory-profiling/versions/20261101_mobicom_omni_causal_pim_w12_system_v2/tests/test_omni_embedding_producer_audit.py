from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_embedding_producers import audit_producers
from omni_embedding_producer import resolve_producers
from omni_embedding_store import EdgeEmbeddingStore
from omni_evidence_planner import build_local_hyperedges


def fixture():
    graph = {'person_id': '0', 'atoms': [
        {'atom_id': str(index), 'session_id': 'session', 'ordinal': index, 'role': 'user',
         'content_type': 'text', 'content': str(index)} for index in range(6)]}
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    edges = build_local_hyperedges(['session'], {'session': graph['atoms']}, 2)
    store = EdgeEmbeddingStore(1024)
    rows = []
    for question in ('first', 'second'):
        _, records = resolve_producers(edges, atoms, store,
            lambda texts: [torch.tensor([float(len(text))]) for text in texts], 'execution',
            output_digest=lambda values: 'a' * 64)
        rows.append({'person_id': '0', 'question_id': question, 'retrieved_session_ids': ['session'],
                     'embedding_producers': records, 'hypergraph_evidence_plan': {
                         'embedding_producers': records, 'embedding_over_budget_edge_ids': []}})
    return {'complete': True, 'evaluation_label_files_read': [], 'rows': rows,
        'embedding_state_reuse': store.summary(), 'embedding_state_reuse_enabled': True,
        'embedding_execution_identity': 'execution'}, graph


class ProducerAuditTests(unittest.TestCase):
    def test_original_hypergraph_operands_and_natural_producer_hit_close(self):
        run, graph = fixture()
        result = audit_producers(run, graph, 4)
        self.assertEqual(result['producer_invocations'], 2)
        self.assertEqual(result['reused_producers'], 1)
        self.assertEqual(result['computed_vectors'], result['reused_vectors'])

    def test_input_identity_counter_or_residency_forgery_is_rejected(self):
        for mutation in ('operand', 'identity', 'hash', 'counter', 'first_hit', 'skip'):
            run, graph = deepcopy(fixture())
            if mutation == 'operand':
                graph['atoms'][0]['content'] = 'changed'
            elif mutation == 'identity':
                run['embedding_execution_identity'] = 'different'
            elif mutation == 'hash':
                run['rows'][1]['embedding_producers'][0]['output_sha256'] = 'b' * 64
            elif mutation == 'counter':
                run['embedding_state_reuse']['resident_bytes'] += 4
            elif mutation == 'first_hit':
                run['rows'][0]['embedding_producers'][0]['reused'] = True
            else:
                run['rows'][0]['embedding_producers'].clear()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_producers(run, graph, 4)


if __name__ == '__main__':
    unittest.main()
