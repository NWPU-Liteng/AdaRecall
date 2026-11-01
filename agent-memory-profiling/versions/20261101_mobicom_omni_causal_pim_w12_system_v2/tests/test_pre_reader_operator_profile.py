import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from omni_proof_runtime import _profiled_operator_events
from residual_profile import ReaderOperatorTrace


class ProfileEvent:
    def __init__(self, key, count, input_shapes, device_time, self_device_time,
                 cpu_time, self_cpu_time):
        self.key = key
        self.count = count
        self.input_shapes = input_shapes
        self.device_time_total = device_time
        self.self_device_time_total = self_device_time
        self.cpu_time_total = cpu_time
        self.self_cpu_time_total = self_cpu_time


class Profile:
    def __init__(self, events):
        self.events = events

    def key_averages(self, *, group_by_input_shape):
        if not group_by_input_shape:
            raise AssertionError('operator shapes must be grouped')
        return self.events


class PreReaderOperatorProfileTests(unittest.TestCase):
    def test_linear_trace_reports_decode_gemv_shapes_without_tensor_values(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 2, bias=False)

            def forward(self, values):
                return self.projection(values)

        model = TinyModel()
        trace = ReaderOperatorTrace(model)
        trace.phase = 'decode'
        try:
            model(torch.ones(1, 3))
        finally:
            trace.close()

        result = trace.as_dict()
        self.assertEqual(result['decode_single_row_linear_calls'], 1)
        self.assertEqual(result['decode_single_row_unique_weight_bytes'], 6 * 4)
        self.assertEqual(result['decode_single_row_macs'], 6)
        self.assertNotIn('activation', result)

    def test_operator_events_keep_shapes_times_and_phase_without_summing_nested_times(self):
        profile = Profile([
            ProfileEvent('aten::addmm', 4, [[1, 64], [64, 128], [128]], 20., 12., 30., 8.),
            ProfileEvent('aten::layer_norm', 4, [[1, 128]], 0., 0., 5., 5.),
        ])

        result = _profiled_operator_events(profile, phase='decode', profiled_forward_calls=4)

        self.assertEqual(result['phase'], 'decode')
        self.assertEqual(result['profiled_model_forward_calls'], 4)
        self.assertEqual(len(result['operator_events']), 2)
        self.assertEqual(result['operator_events'][0]['operator'], 'aten::addmm')
        self.assertEqual(result['operator_events'][0]['input_shapes'], [[1, 64], [64, 128], [128]])
        self.assertEqual(result['operator_events'][0]['inclusive_device_microseconds'], 20.)
        self.assertIn('may overlap', result['timing_scope'])


if __name__ == '__main__':
    unittest.main()
