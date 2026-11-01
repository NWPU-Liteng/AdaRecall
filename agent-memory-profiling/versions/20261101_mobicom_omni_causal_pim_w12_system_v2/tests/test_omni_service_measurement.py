from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_service_measurement import measure_vision, service_stage_ledger


class Model:
    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        if kwargs.get('fail'):
            raise ValueError('expected failure')
        return pixel_values + image_grid_thw


class ServiceMeasurementTests(unittest.TestCase):
    def test_nested_vision_is_not_added_to_disjoint_total(self):
        result = service_stage_ledger(12., evidence_seconds=2., input_preparation_seconds=1.,
            prefill_seconds=4., decode_seconds=3., answer_conversion_seconds=1.,
            vision_in_prefill_seconds=2.)
        self.assertEqual(sum(result['disjoint_seconds'].values()), 12.)
        self.assertEqual(result['nested_seconds']['language_and_wrapper_prefill_seconds'], 2.)
        with self.assertRaises(ValueError):
            service_stage_ledger(12., evidence_seconds=2., input_preparation_seconds=1.,
                prefill_seconds=4., decode_seconds=3., answer_conversion_seconds=1.,
                vision_in_prefill_seconds=5.)

    def test_wrapper_preserves_result_and_restores_even_after_error(self):
        model = Model()
        synchronize = lambda: None
        with measure_vision(model, synchronize) as measurement:
            self.assertEqual(model.get_image_features(2, 3), 5)
        self.assertEqual(measurement['vision_calls'], 1)
        self.assertNotIn('get_image_features', model.__dict__)
        with self.assertRaises(ValueError):
            with measure_vision(model, synchronize):
                model.get_image_features(2, 3, fail=True)
        self.assertNotIn('get_image_features', model.__dict__)


if __name__ == '__main__':
    unittest.main()
