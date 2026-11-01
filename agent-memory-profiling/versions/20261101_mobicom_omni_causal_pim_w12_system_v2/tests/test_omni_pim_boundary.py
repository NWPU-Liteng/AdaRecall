import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_pim_boundary import empirical, required_fraction
from omni_decode_lifecycle import normalized_service_sensitivity


class PimBoundaryTests(unittest.TestCase):
    def test_dimensionless_sensitivity_matches_noop_and_full_decode_limits(self):
        self.assertEqual(normalized_service_sensitivity(.8, 1.5, 0.)['conditional_service_speedup'], 1.)
        result = normalized_service_sensitivity(.8, 1.5, 1.)
        self.assertAlmostEqual(result['conditional_service_speedup'], 1 / (.2 + .8 / 1.5))

    def test_required_fraction_is_not_clipped_or_presented_as_guarantee(self):
        value = required_fraction(.886932872853126, 1.7660982211008143, 1.3)
        self.assertAlmostEqual(value, .5998153144223249)
        self.assertGreater(required_fraction(.886932872853126, 1.7660982211008143, 1.5), value)
        self.assertLess(value, 1.)
        self.assertGreater(required_fraction(.2, 1.2, 1.5), 1.)
        for target in (1.3, 1.5):
            fraction = required_fraction(.886932872853126, 1.7660982211008143, target)
            self.assertAlmostEqual(normalized_service_sensitivity(
                .886932872853126, 1.7660982211008143, fraction)['conditional_service_speedup'], target)

    def test_empirical_keeps_full_denominator_and_tail(self):
        result = empirical([1., 2., 3., 4.])
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['p50_nearest_rank_seconds'], 2.)
        self.assertEqual(result['p95_nearest_rank_seconds'], 4.)
        with self.assertRaises(ValueError):
            empirical([1., 0.])

    def test_invalid_target_and_speedup_fail_closed(self):
        with self.assertRaises(ValueError):
            required_fraction(.8, 1., 1.3)
        with self.assertRaises(ValueError):
            required_fraction(.8, 1.5, 1.)
        for fraction in (0., -1., math.nan):
            with self.assertRaises(ValueError):
                required_fraction(fraction, 1.5, 1.3)


if __name__ == '__main__':
    unittest.main()
