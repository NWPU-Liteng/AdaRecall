import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ReuseLauncherTests(unittest.TestCase):
    def check_rejected_mapping(self, policies, offset):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'nvidia-smi'
            executable.write_text('#!/bin/sh\nprintf "0\\n1\\n2\\n3\\n4\\n5\\n6\\n"\n')
            executable.chmod(0o755)
            output = root / 'run'
            environment = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'],
                               RUN_ROOT=str(output), SCREEN_POLICIES=policies, GPU_OFFSET=str(offset))
            script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
            result = subprocess.run(['bash', str(script)], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('invalid or duplicate GPU assignment', result.stderr)
            self.assertFalse(output.exists())

    def test_expanded_offset_is_validated_before_launch(self):
        self.check_rejected_mapping('expanded', 3)

    def test_overlapping_policy_devices_are_rejected(self):
        self.check_rejected_mapping('expanded milp', 0)

    def test_invalid_image_quota_is_rejected_before_accessing_data(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
        result = subprocess.run(['bash', str(script)], env=dict(os.environ, IMAGE_EDGES_PER_SESSION='-1'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_invalid_token_cost_reuse_flag_is_rejected(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
        result = subprocess.run(['bash', str(script)], env=dict(os.environ, QUERY_LOCAL_TOKEN_COST_REUSE='yes'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_invalid_deferred_finite_check_flag_is_rejected(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
        result = subprocess.run(['bash', str(script)], env=dict(os.environ, DEFER_INT8_FINITE_CHECKS='yes'),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_deferred_finite_checks_require_w8a8(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
        result = subprocess.run(['bash', str(script)], env=dict(os.environ,
            DEFER_INT8_FINITE_CHECKS='1', READER_QUANTIZATION='bf16'),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('deferred finite-check aggregation requires shared W8A8 primary', result.stderr)

    def test_pimnast_requires_int8_and_existing_projection(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/run_reuse_qualification.sh'
        result = subprocess.run(['bash', str(script)], env=dict(os.environ,
            PIMNAST_FUNCTIONAL_PROJECTION='/not/a/projection.json', READER_QUANTIZATION='bf16'),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
