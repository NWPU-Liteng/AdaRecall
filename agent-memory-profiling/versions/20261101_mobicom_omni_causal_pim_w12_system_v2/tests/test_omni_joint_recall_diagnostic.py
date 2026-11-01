from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_omni_joint_recall import coverage


class JointRecallDiagnosticTests(unittest.TestCase):
    def test_source_and_image_coverage_are_distinct_and_unresolved_is_not_a_hit(self):
        atoms = [dict(session_id='s', image_refs=[dict(status='ambiguous', normalized='wrong'),
                 dict(status='exact', normalized='good')])]
        result = coverage({'s'}, {'good', 'wrong'}, atoms)
        self.assertEqual(result['source_session_recall'], 1.)
        self.assertEqual(result['image_recall'], .5)
        self.assertEqual(result['missing_images'], ['wrong'])
        self.assertIsNone(coverage(set(), set(), atoms)['image_recall'])


if __name__ == '__main__':
    unittest.main()
