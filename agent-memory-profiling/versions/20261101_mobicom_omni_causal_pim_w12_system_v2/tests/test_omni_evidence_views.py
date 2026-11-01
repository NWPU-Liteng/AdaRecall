import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_evidence_views import build_evidence_views, produce_evidence_views
from omni_evidence_compiler import compile_evidence
from omni_joint_index import JointIndex, digest


class EvidenceViewTests(unittest.TestCase):
    def fixture(self):
        atoms = {
            't': {'atom_id': 't', 'ordinal': 0, 'session_id': 's0', 'role': 'user',
                  'content_type': 'text', 'content': 'plain', 'image_refs': []},
            'u': {'atom_id': 'u', 'ordinal': 1, 'session_id': 's1', 'role': 'user',
                  'content_type': 'text', 'content': 'image text', 'image_refs': [
                      {'status': 'exact', 'normalized': 'uid0/pic.jpg',
                       'resolved': 'uid0/pic.jpg'}]},
        }
        edges = [
            {'edge_id': 'text-edge', 'atom_ids': ['t'], 'embedding_text': 'plain'},
            {'edge_id': 'image-edge', 'atom_ids': ['u'], 'embedding_text': 'image text'},
        ]
        return edges, atoms

    def test_views_are_independent_and_keep_exchange_provenance(self):
        edges, atoms = self.fixture()
        views = build_evidence_views(edges, atoms)
        self.assertEqual([view['view_type'] for view in views], ['text', 'text', 'image'])
        image_view = views[-1]
        self.assertEqual(image_view['edge_id'], 'image-edge')
        self.assertEqual(image_view['atom_ids'], ['u'])
        self.assertEqual(image_view['source_atom_id'], 'u')
        self.assertEqual(image_view['image_identity'], 'uid0/pic.jpg')

    def test_text_only_producer_reused_and_only_missing_views_encoded(self):
        edges, atoms = self.fixture()
        calls = []

        def encode(documents):
            calls.extend(documents)
            return np.asarray([[1., 0.] if doc['text'] else [0., 1.] for doc in documents]), [{}] * len(documents)

        views, vectors, stats = produce_evidence_views(
            edges, atoms, np.asarray([[0., 1.], [.7, .7]], dtype=np.float32), encode,
            lambda ref: ('pixels', 'image-hash'), batch_size=4)
        self.assertEqual(len(calls), 2)
        self.assertEqual(stats['encoded_producers'], 2)
        np.testing.assert_array_equal(vectors[0], [0., 1.])
        np.testing.assert_array_equal(vectors[1], [1., 0.])
        np.testing.assert_array_equal(vectors[2], [0., 1.])
        self.assertEqual(views[-1]['producer']['image_sha256'], 'image-hash')

    def test_index_searches_independent_image_view_but_returns_parent_hyperedge(self):
        edges, atoms = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            edge_vectors = np.asarray([[0., 1.], [1., 0.]], dtype=np.float32)
            np.save(root / 'vectors.npy', edge_vectors)
            views = [
                {'view_id': 'text:text-edge', 'edge_id': 'text-edge', 'view_type': 'text',
                 'atom_ids': ['t']},
                {'view_id': 'text:image-edge', 'edge_id': 'image-edge', 'view_type': 'text',
                 'atom_ids': ['u']},
                {'view_id': 'image:image-edge:u', 'edge_id': 'image-edge', 'view_type': 'image',
                 'atom_ids': ['u'], 'source_atom_id': 'u', 'image_identity': 'uid0/pic.jpg',
                 'image_ref': {'status': 'exact', 'normalized': 'uid0/pic.jpg',
                               'resolved': 'uid0/pic.jpg'}},
            ]
            np.save(root / 'evidence_view_vectors.npy',
                    np.asarray([[0., 1.], [.8, .6], [1., 0.]], dtype=np.float32))
            metadata = {'complete': True, 'evaluation_label_files_read': [],
                'prepared_audit_sha256': 'audit', 'person_id': '0',
                'model_files_sha256': {'config': 'model'}, 'image_max_pixels': 65536,
                'archive_sha256': 'archive', 'vectors_sha256': digest(root / 'vectors.npy'),
                'evidence_view_vectors_sha256': digest(root / 'evidence_view_vectors.npy'),
                'edges': edges, 'dimension': 2, 'evidence_views': views}
            (root / 'index.json').write_text(json.dumps(metadata))
            index = JointIndex(root, 'audit', '0', {'config': 'model'}, 65536, 'archive')
            selected, scores, vectors, records = index.search_evidence_units(np.array([1., 0.]), 1)
        self.assertEqual(selected[0]['edge_id'], 'image-edge')
        self.assertEqual(records[0]['view_type'], 'image')
        self.assertAlmostEqual(scores[0], 1.)
        np.testing.assert_array_equal(vectors[0], [1., 0.])

    def test_rejects_unbound_or_invalid_image_view(self):
        edges, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'vectors.npy', np.asarray([[0., 1.], [1., 0.]], dtype=np.float32))
            np.save(root / 'evidence_view_vectors.npy', np.asarray([[0., 1.]], dtype=np.float32))
            metadata = {'complete': True, 'evaluation_label_files_read': [],
                'prepared_audit_sha256': 'audit', 'person_id': '0',
                'model_files_sha256': {'config': 'model'}, 'image_max_pixels': 65536,
                'archive_sha256': 'archive', 'vectors_sha256': digest(root / 'vectors.npy'),
                'evidence_view_vectors_sha256': digest(root / 'evidence_view_vectors.npy'),
                'edges': edges, 'dimension': 2, 'evidence_views': [
                    {'view_id': 'bad', 'edge_id': 'missing', 'view_type': 'image', 'atom_ids': []}]}
            (root / 'index.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                JointIndex(root, 'audit', '0', {'config': 'model'}, 65536, 'archive')

    def test_shared_image_is_encoded_once_but_keeps_each_observed_parent(self):
        edges, atoms = self.fixture()
        atoms['v'] = {**atoms['u'], 'atom_id': 'v', 'ordinal': 2}
        edges.append({'edge_id': 'other-image-parent', 'atom_ids': ['v'], 'embedding_text': 'other text'})
        calls = []

        def encode(documents):
            calls.extend(documents)
            return np.asarray([[1., 0.]] * len(documents)), [{}] * len(documents)

        views, _, stats = produce_evidence_views(
            edges, atoms, np.asarray([[1., 0.]] * 3), encode,
            lambda ref: ('pixels', 'image-hash'))
        self.assertEqual(sum(bool(document['images']) for document in calls), 1)
        self.assertEqual(stats['encoded_producers'], 3)
        self.assertEqual(len([view for view in views if view['view_type'] == 'image']), 2)

    def test_max_pool_is_not_a_sum_reward_for_more_views(self):
        index = JointIndex.__new__(JointIndex)
        index.edges = [{'edge_id': 'a'}, {'edge_id': 'b'}]
        index.edge_positions = {'a': 0, 'b': 1}
        index.hits = 0
        index.views = [
            {'view_id': 'a:text', 'edge_id': 'a', 'view_type': 'text'},
            {'view_id': 'a:image', 'edge_id': 'a', 'view_type': 'image'},
            {'view_id': 'b:text', 'edge_id': 'b', 'view_type': 'text'},
        ]
        index.view_vectors = np.asarray([[.6, .8], [.6, .8], [.8, .6]], dtype=np.float32)
        selected, scores, vectors, records = index.search_evidence_units([1., 0.], 2)
        self.assertEqual([edge['edge_id'] for edge in selected], ['b', 'a'])
        np.testing.assert_allclose(scores, [.8, .6])
        self.assertEqual(index.hits, 2)
        self.assertEqual(records[1]['view_id'], 'a:image')
        np.testing.assert_allclose(vectors @ [1., 0.], scores)
        with self.assertRaises(ValueError):
            index.search_evidence_units([0., 0.], 1)

    def test_image_trigger_keeps_full_exchange_and_actual_image_budget(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

        edges, atoms = self.fixture()
        vectors = np.asarray([[0., 1.], [1., 0.]])
        selected = compile_evidence(edges, vectors, [1., 0.], atoms, Tokenizer(),
            {'uid0/pic.jpg': 1024}, text_budget=20, max_images=1, input_budget=1100,
            fixed_prompt_tokens=1, time_limit=1.)
        self.assertEqual(selected['accepted_atom_ids'], ['u'])
        self.assertEqual(selected['images'][0]['source_atom_id'], 'u')
        self.assertEqual(selected['visual_tokens_from_processor_grid'], 1024)
        rejected = compile_evidence(edges, vectors, [1., 0.], atoms, Tokenizer(),
            {'uid0/pic.jpg': 1024}, text_budget=20, max_images=1, input_budget=1000,
            fixed_prompt_tokens=1, time_limit=1.)
        self.assertNotIn('u', rejected['accepted_atom_ids'])
        self.assertEqual(rejected['images'], [])


if __name__ == '__main__':
    unittest.main()
