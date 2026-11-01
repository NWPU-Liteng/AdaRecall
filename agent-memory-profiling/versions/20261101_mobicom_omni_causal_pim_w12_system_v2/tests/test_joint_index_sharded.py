import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_omni_joint_index_sharded import atomic_json, merge, validate_batch
from omni_joint_index import digest


class ShardedJointIndexTests(unittest.TestCase):
    def record(self, position=0):
        return {'batch_index': position, 'start': position, 'end': position + 1,
                'edge_ids': [f'edge_{position}'],
                'producers': [{'edge_id': f'edge_{position}'}], 'construction_seconds': 2.}

    def test_checkpoint_rejects_changed_producer_order(self):
        record = self.record()
        record['producers'][0]['edge_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity'):
            validate_batch(record, np.array([[1., 0.]], dtype=np.float32),
                           [{'edge_id': 'edge_0'}], 0, 1)

    def test_checkpoint_rejects_nonfinite_nonunit_or_low_precision(self):
        for values in (np.array([[float('nan'), 0.]], dtype=np.float32),
                       np.array([[2., 0.]], dtype=np.float32),
                       np.array([[1., 0.]], dtype=np.float16)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, 'vectors'):
                    validate_batch(self.record(), values, [{'edge_id': 'edge_0'}], 0, 1)

    def fixture(self, root):
        prepared = root / 'prepared'
        prepared.mkdir()
        (prepared / 'audit.json').write_text('{}')
        edges = [{'edge_id': f'edge_{index}'} for index in range(2)]
        shards = [root / f'shard_{index}' for index in range(2)]
        for position, directory in enumerate(shards):
            (directory / 'batches').mkdir(parents=True)
            vector_path = directory / 'batches' / f'batch_{position:06d}.npy'
            np.save(vector_path, np.array([[1., 0.]], dtype=np.float32))
            atomic_json(vector_path.with_suffix('.json'),
                        {**self.record(position), 'vectors_sha256': digest(vector_path)})
            state = {'complete': True, 'person_id': '0',
                     'prepared_audit_sha256': digest(prepared / 'audit.json'),
                     'archive_sha256': 'archive', 'model_files_sha256': {'weight': 'model'},
                     'image_max_pixels': 65536, 'max_input_tokens': 8192, 'batch_size': 1,
                     'candidate_edges': 2, 'module_sha256': 'module', 'script_sha256': 'script',
                     'rejected_edges': [], 'evaluation_label_files_read': [],
                     'edge_scope': 'image_bearing_complete_exchange', 'pooling': 'lasttoken',
                     'multi_image_extension': 'all', 'visibility_protocol': 'history_end',
                     'shard_count': 2, 'shard_index': position, 'total_batches': 2,
                     'batch_indices': [position], 'construction_seconds': 2.,
                     'initialization_seconds_sum': 1., 'incomplete_attempts': 0,
                     'peak_cuda_allocated_bytes': 123,
                     'attempts': [{'complete': True, 'started_unix_seconds': 101. + position,
                                  'ended_unix_seconds': 105. + position,
                                  'initialization_seconds': 1.}]}
            atomic_json(directory / 'shard.json', state)
        args = argparse.Namespace(prepared=prepared, image_archive=root / 'image.zip',
                                  output=root / 'merged', user_id='0', batch_size=1,
                                  image_max_pixels=65536, max_input_tokens=8192,
                                  image_only=True, shard_count=2, shards=shards)
        return args, edges

    def invoke_merge(self, args, edges):
        with patch('build_omni_joint_index_sharded.source_state', return_value=({}, edges, [], {}, 'archive')):
            merge(args)

    def test_merge_preserves_order_and_counts_all_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            args, edges = self.fixture(Path(directory))
            args.shards.reverse()
            self.invoke_merge(args, edges)
            state = json.loads((args.output / 'index.json').read_text())
            self.assertEqual(state['edges'], edges)
            self.assertEqual([row['edge_id'] for row in state['producers']], ['edge_0', 'edge_1'])
            self.assertEqual(state['construction_seconds'], 4.)
            self.assertEqual(state['construction_wall_seconds'], 6.)

    def test_merge_rejects_mixed_models_and_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            args, edges = self.fixture(Path(directory))
            state_path = args.shards[1] / 'shard.json'
            state = json.loads(state_path.read_text())
            state['model_files_sha256'] = {'weight': 'different'}
            atomic_json(state_path, state)
            with self.assertRaisesRegex(ValueError, 'identity'):
                self.invoke_merge(args, edges)
            args.shards = args.shards[:1]
            with self.assertRaisesRegex(ValueError, 'shard set'):
                self.invoke_merge(args, edges)

    def test_merge_rejects_corrupt_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            args, edges = self.fixture(Path(directory))
            np.save(args.shards[0] / 'batches' / 'batch_000000.npy',
                    np.array([[0., 1.]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                self.invoke_merge(args, edges)

    def test_merge_rejects_different_processor_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            args, edges = self.fixture(Path(directory))
            args.image_max_pixels = 16384
            with self.assertRaisesRegex(ValueError, 'protocol'):
                self.invoke_merge(args, edges)


if __name__ == '__main__':
    unittest.main()
