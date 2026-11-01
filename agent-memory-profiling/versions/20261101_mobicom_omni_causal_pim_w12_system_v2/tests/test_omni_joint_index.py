import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_candidate_union import candidate_union
from omni_joint_index import JointIndex, TrainedMultimodalReranker, digest, rank_indices


class JointIndexTests(unittest.TestCase):
    def test_unknown_patch_projection_fails_before_model_loading(self):
        with patch('omni_joint_index.import_official') as loader:
            with self.assertRaisesRegex(ValueError, 'patch_projection'):
                TrainedMultimodalReranker(Path('unused'), 'cpu', 65536, patch_projection='unknown')
            loader.assert_not_called()

    def test_linear_patch_keeps_weights_and_matches_cpu_projection(self):
        torch.manual_seed(23)
        projection = torch.nn.Conv3d(3, 5, (2, 3, 3), stride=(2, 3, 3)).double()
        inputs = torch.randn(7, 3, 2, 3, 3, dtype=torch.float64)
        reference = projection(inputs).flatten(1)
        original_weight = projection.weight.detach().clone()
        wrapper = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(
            patch_embed=SimpleNamespace(proj=projection))), processor=SimpleNamespace(
                image_processor=SimpleNamespace(size={}), tokenizer=SimpleNamespace()))
        with patch('omni_joint_index.import_official', return_value=SimpleNamespace(
                Qwen3VLReranker=Mock(return_value=wrapper))):
            reranker = TrainedMultimodalReranker(Path('unused'), 'cpu', 65536, patch_projection='linear')
        self.assertTrue(torch.equal(projection.weight, original_weight))
        self.assertTrue(torch.allclose(projection(inputs).flatten(1), reference, atol=1e-12, rtol=1e-12))
        self.assertEqual(reranker.patch_projection, 'linear')

    def test_default_patch_keeps_original_forward(self):
        projection = torch.nn.Conv3d(3, 5, 2, stride=2)
        original_forward = projection.forward
        wrapper = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(
            patch_embed=SimpleNamespace(proj=projection))), processor=SimpleNamespace(
                image_processor=SimpleNamespace(size={}), tokenizer=SimpleNamespace()))
        with patch('omni_joint_index.import_official', return_value=SimpleNamespace(
                Qwen3VLReranker=Mock(return_value=wrapper))):
            reranker = TrainedMultimodalReranker(Path('unused'), 'cpu', 65536)
        self.assertEqual(projection.forward, original_forward)
        self.assertEqual(reranker.patch_projection, 'conv')

    def reranker_fixture(self):
        class Inputs(dict):
            def __getattr__(self, key):
                return self[key]

            def to(self, device):
                return self

        processor = Mock(return_value=Inputs(input_ids=torch.tensor([[1, 2], [1, 0]]),
                                             attention_mask=torch.tensor([[1, 1], [1, 0]])))
        processor.apply_chat_template.return_value = 'prompt'
        class Model:
            def get_image_features(self, pixels, grid=None, **kwargs):
                return pixels

            def __call__(self, **kwargs):
                self.get_image_features(torch.ones(1))
                return SimpleNamespace(last_hidden_state=torch.ones(2, 2, 1))

        wrapper = SimpleNamespace(processor=processor, default_instruction='relevance',
            format_mm_instruction=lambda *args, **kwargs: [{}, {'content': [{'text': '\n<Document>:'}]}],
            model=Model(),
            score_linear=lambda hidden: hidden)
        reranker = TrainedMultimodalReranker.__new__(TrainedMultimodalReranker)
        reranker.device, reranker.profile, reranker.patch_projection = torch.device('cpu'), True, 'conv'
        reranker.batch_size, reranker.max_input_tokens, reranker.wrapper = 2, 10, wrapper
        atoms = {'a': {'ordinal': 0, 'image_refs': []}, 'b': {'ordinal': 1, 'image_refs': [
            {'status': 'exact', 'normalized': 'image.png'}], 'session_id': 's', 'atom_id': 'b'}}
        edges = [{'edge_id': identity, 'atom_ids': [identity], 'embedding_text': identity}
                 for identity in ('a', 'b')]
        return reranker, edges, atoms

    def test_profile_preserves_scores_and_accounts_for_phase_time(self):
        reranker, edges, atoms = self.reranker_fixture()
        scores, records, timings = reranker.score('query', edges, atoms, lambda ref: ('image', 'sha'))
        self.assertEqual(scores, [1., 1.])
        self.assertEqual([record['edge_id'] for record in records], ['a', 'b'])
        self.assertEqual(records[1]['image_sha256'], ['sha'])
        self.assertEqual(timings['candidate_count'], 2)
        self.assertEqual(timings['batch_count'], 1)
        self.assertEqual(timings['image_count'], 1)
        self.assertEqual(timings['padded_input_tokens'], 4)
        self.assertEqual(timings['valid_input_tokens'], 3)
        self.assertEqual(timings['vision_calls'], 1)
        self.assertEqual(len(timings['batches']), 1)
        self.assertFalse(timings['synchronized_cuda'])
        self.assertGreaterEqual(timings['residual_seconds'], 0.)
        self.assertGreater(timings['total_seconds'], 0.)
        self.assertEqual(timings['patch_projection'], 'conv')

    def test_profile_keeps_complete_input_budget_guard(self):
        reranker, edges, atoms = self.reranker_fixture()
        reranker.max_input_tokens = 1
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            reranker.score('query', edges, atoms, lambda ref: ('image', 'sha'))

    def test_cosine_rank_ties_use_identity_not_source_order(self):
        positions, scores = rank_indices(np.array([[1., 0.], [1., 0.], [0., 1.]]),
                                        np.array([1., 0.]), ['z', 'a', 'b'], 2)
        self.assertEqual(positions, [1, 0])
        self.assertEqual(scores, [1., 1.])

    def test_nonfinite_vectors_fail_closed(self):
        with self.assertRaises(ValueError):
            rank_indices(np.array([[float('nan')]]), np.array([1.]), ['a'], 1)

    def test_candidate_union_never_adds_incomparable_scores(self):
        visual = [{'edge_id': 'a', 'atom_ids': ['a']}, {'edge_id': 'b', 'atom_ids': ['b']}]
        text = [{'edge_id': 'b', 'atom_ids': ['b']}, {'edge_id': 'c', 'atom_ids': ['c']}]
        edges, records = candidate_union(visual, [.9, .8], text, [3., 7.], 2)
        self.assertEqual([edge['edge_id'] for edge in edges], ['a', 'b', 'c'])
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-2]['score_semantics'], 'text_reranker_logit_difference')

    def test_candidate_identity_collision_rejected(self):
        with self.assertRaises(ValueError):
            candidate_union([{'edge_id': 'a', 'atom_ids': ['a']}], [1.],
                            [{'edge_id': 'a', 'atom_ids': ['b']}], [1.], 1)

    def test_index_binding_rejects_modified_payload_and_user(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'vectors.npy', np.array([[1., 0.]], dtype=np.float32))
            metadata = {'complete': True, 'evaluation_label_files_read': [],
                        'prepared_audit_sha256': 'audit', 'person_id': '0',
                        'model_files_sha256': {'config': 'model'}, 'image_max_pixels': 65536,
                        'archive_sha256': 'archive', 'vectors_sha256': digest(root / 'vectors.npy'),
                        'edges': [{'edge_id': 'a'}], 'dimension': 2}
            (root / 'index.json').write_text(json.dumps(metadata))
            index = JointIndex(root, 'audit', '0', {'config': 'model'}, 65536, 'archive')
            self.assertEqual(index.search(np.array([1., 0.]), 1)[1], [1.])
            with self.assertRaises(ValueError):
                JointIndex(root, 'audit', '1', {'config': 'model'}, 65536, 'archive')
            np.save(root / 'vectors.npy', np.array([[0., 1.]], dtype=np.float32))
            with self.assertRaises(ValueError):
                JointIndex(root, 'audit', '0', {'config': 'model'}, 65536, 'archive')


if __name__ == '__main__':
    unittest.main()
