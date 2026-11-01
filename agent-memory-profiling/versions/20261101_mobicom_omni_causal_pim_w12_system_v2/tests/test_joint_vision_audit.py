import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_joint_vision_partition import audit


class JointVisionAuditTests(unittest.TestCase):
    def fixture(self, root):
        source_dir = root / 'sources'
        source_dir.mkdir()
        (source_dir / 'probe.py').write_text('frozen')
        comparison = {'finite': True, 'same_shape_dtype': True, 'bitwise_equal': False,
                      'max_absolute_difference': 1.0}
        result = {'complete': True, 'question_id': '0_q', 'source_run_sha256': 'request',
            'evaluation_label_files_read': [], 'image_sha256': ['first', 'second'],
            'image_grid_thw': [[1, 2, 2], [1, 2, 2]],
            'pooled_feature_comparisons': [comparison, comparison], 'deepstack_comparisons': [comparison],
            'all_features_bitwise_equal': False, 'mainline_promotion_authorized': False,
            'main_experiment_closed': False, 'actual_pim_end_to_end_speedup': None,
            'reference_profiled_seconds': 1.0}
        registration = {'question_id': '0_q', 'source_run_sha256': 'request',
            'sources_sha256': {'probe.py': hashlib.sha256(b'frozen').hexdigest()}}
        result_path, registration_path = root / 'result.json', root / 'registration.json'
        result_path.write_text(json.dumps(result))
        registration_path.write_text(json.dumps(registration))
        return result_path, registration_path, source_dir

    def test_numerical_rejection_is_a_valid_diagnostic_not_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            report = audit(*self.fixture(Path(directory)))
            self.assertTrue(report['complete'])
            self.assertFalse(report['exact_feature_reuse_qualified'])
            self.assertFalse(report['main_experiment_closed'])

    def test_tampered_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            (paths[2] / 'probe.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
                audit(*paths)

    def test_inconsistent_equality_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            result = json.loads(paths[0].read_text())
            result['all_features_bitwise_equal'] = True
            paths[0].write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                audit(*paths)


if __name__ == '__main__':
    unittest.main()
