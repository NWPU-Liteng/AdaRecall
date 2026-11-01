import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_development_arm as arm


class RunDevelopmentArmTests(unittest.TestCase):
    def test_w12_command_is_fully_frozen(self):
        registration = {'arguments': {'config': 'config.yaml', 'seed': 7}}
        cohort = {'question_ids_by_user': {'0': [f'0_q_{i}' for i in range(64)]}}
        command = arm.command_for(
            registration, cohort, Path('cohort.json'), '0',
            Path('baseline_user_0.json'), 'w12a16', 12, 128)
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith('scripts/run_omni_raw_image_smoke.py'))
        self.assertIn('--reader-quantization', command)
        self.assertEqual(command[command.index('--reader-quantization') + 1], 'w12a16')
        self.assertEqual(command[command.index('--w12a16-group-size') + 1], '128')
        self.assertEqual(command[command.index('--limit') + 1], '64')
        self.assertEqual(command[command.index('--device') + 1], 'cuda:0')
        self.assertEqual(command[command.index('--hardware-identity') + 1],
                         'RTX3090:development-192-arm')

    def test_digest_streams_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'payload.bin'
            payload = b'w12a16\n' * 10000
            path.write_bytes(payload)
            self.assertEqual(arm.digest(path), hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
