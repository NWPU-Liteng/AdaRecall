import math
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

from eval_conformal_coverage import build_coverage_report, summarize_coverage
from fit_calibration import build_calibrator, write_calibrator
from test_fit_calibration import create_fixture


class ConformalCoverageTests(unittest.TestCase):
    def test_all_candidate_pool_misses_are_not_counted_as_covered(self):
        calibrator = {
            'alphas': [0.1, 0.2],
            'tau': {'0.1': 0.8, '0.2': None},
        }
        summary = summarize_coverage(
            [0.9, 0.7, -math.inf], calibrator)

        self.assertEqual(summary['n'], 3)
        self.assertEqual(summary['pool_miss_count'], 1)
        self.assertEqual(summary['pool_miss_ratio'], 1 / 3)
        self.assertEqual(summary['coverage']['0.1']['covered'], 1)
        self.assertEqual(summary['coverage']['0.1']['empirical_coverage'], 1 / 3)
        self.assertEqual(summary['coverage']['0.1']['target_coverage'], 0.9)
        self.assertEqual(summary['coverage']['0.2']['tau'], None)
        self.assertEqual(summary['coverage']['0.2']['covered'], 2)
        self.assertEqual(summary['coverage']['0.2']['empirical_coverage'], 2 / 3)

    def test_empty_evidence_sample_reports_null_coverage(self):
        summary = summarize_coverage([], {'alphas': [0.1], 'tau': {'0.1': 0.0}})
        self.assertEqual(summary['n'], 0)
        self.assertIsNone(summary['pool_miss_ratio'])
        self.assertIsNone(
            summary['coverage']['0.1']['empirical_coverage'])

    def test_report_reads_only_validation_dumps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, dump = create_fixture(root)
            calibrator = build_calibrator(dump, labels)
            calibrator_path = write_calibrator(
                calibrator, root / 'calibrator.json')
            for path in dump.glob('*/*.json'):
                payload = json.loads(path.read_text())
                payload['split'] = 'validation'
                path.write_text(json.dumps(payload))
            report = build_coverage_report(dump, labels, calibrator_path)

        self.assertEqual(report['schema'], 'omni-conformal-coverage-v1')
        self.assertEqual(report['split'], 'validation')
        self.assertEqual(report['questions_total'], 2)
        self.assertEqual(report['pool_miss_count'], 1)
        self.assertEqual(report['pool_miss_ratio'], 0.5)
        self.assertEqual(report['coverage']['0.1']['n'], 2)
        self.assertEqual(report['coverage']['0.1']['covered'], 1)
        self.assertEqual(report['coverage']['0.1']['empirical_coverage'], 0.5)


if __name__ == '__main__':
    unittest.main()
