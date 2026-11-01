import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_hyperpim_sc_holdout200 as runner


class HoldoutW12CommandTests(unittest.TestCase):
    def protocol(self, group_size=128):
        arm = {
            'reader_quantization': 'w12a16',
            'w8a16_group_size': 128,
            'w8a16_residual_bits': 0,
            'w8a16_fallback_scope': 'none',
            'w12a16_group_size': group_size,
        }
        if group_size is None:
            del arm['w12a16_group_size']
        return {
            'cohort': {'path': 'cohort.json'},
            'common_arguments': {
                'prepared': 'prepared', 'image_archive': 'image.zip',
                'reader': 'reader', 'embedder': 'embedder', 'reranker': 'reranker',
                'per_user': 50, 'allowed_splits': ['holdout'],
                'candidate_sessions': 4, 'max_images': 4, 'max_new_tokens': 512,
                'session_text_tokens': 900, 'total_text_tokens': 3600,
                'evidence_policy': 'local_hyperedge', 'image_max_pixels': 262144,
                'planner_time_limit': 5.0, 'concurrency_contract': 'isolated',
                'startup_policy': 'model_load_outside_request_per_worker',
            },
            'arms': {'w12a16': {'arguments': arm}},
        }

    def test_w12_group_size_is_explicit_and_passed_to_serving(self):
        command = runner.command_for(self.protocol(), {'arm': 'w12a16', 'user': '12'},
                                     Path('out.json'))
        self.assertEqual(command[command.index('--reader-quantization') + 1], 'w12a16')
        self.assertEqual(command[command.index('--w12a16-group-size') + 1], '128')
        self.assertIn('--allowed-splits', command)
        self.assertEqual(command[command.index('--allowed-splits') + 1], 'holdout')

    def test_w12_without_frozen_group_size_fails_closed(self):
        with self.assertRaises(ValueError):
            runner.command_for(self.protocol(group_size=None),
                               {'arm': 'w12a16', 'user': '12'}, Path('out.json'))


if __name__ == '__main__':
    unittest.main()
