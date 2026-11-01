import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from profile_omni_int8_decode import trace_kernel_attribution


class DecodeProfileTests(unittest.TestCase):
    def event(self, category, name, start, duration, external=None):
        return {'ph': 'X', 'cat': category, 'name': name, 'pid': 1, 'tid': 1,
                'ts': start, 'dur': duration,
                'args': {} if external is None else {'External id': external}}

    def run_trace(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': events}))
            return trace_kernel_attribution(path)

    def fixture(self):
        return [self.event('user_annotation', 'omni.decode_step', 0, 100),
                self.event('user_annotation', 'omni.int8.linear.selected', 10, 80),
                self.event('user_annotation', 'omni.int8.quantization', 20, 10),
                self.event('user_annotation', 'omni.int8.integer.selected', 40, 10),
                self.event('cpu_op', 'quantize_op', 21, 2, 1),
                self.event('cpu_op', 'multiply_op', 41, 2, 2),
                self.event('cpu_op', 'cast_op', 61, 2, 3),
                self.event('cpu_op', 'other_op', 91, 2, 4),
                self.event('kernel', 'quantize_kernel', 121, 3, 1),
                self.event('kernel', 'integer_kernel', 131, 5, 2),
                self.event('gpu_memcpy', 'cast_copy', 141, 2, 3),
                self.event('kernel', 'other_kernel', 151, 4, 4),
                self.event('gpu_user_annotation', 'omni.int8.quantization', 0, 99999)]

    def test_each_activity_counted_once_using_cpu_owner_not_gpu_timestamp(self):
        result = self.run_trace(list(reversed(self.fixture())))
        self.assertTrue(result['kernel_activity_partition_valid'])
        totals = result['kernel_activity_partition_microseconds']
        self.assertEqual(totals['quantization'], 3)
        self.assertEqual(totals['selected_integer_product'], 5)
        self.assertEqual(totals['linear_remaining'], 2)
        self.assertEqual(totals['nonlinear_and_token'], 4)
        self.assertEqual(sum(totals.values()), 14)
        self.assertAlmostEqual(sum(result['kernel_activity_fractions_of_attributed_decode'].values()), 1.)

    def test_unknown_gpu_work_remains_charged_and_invalidates_complete_attribution(self):
        events = self.fixture() + [self.event('kernel', 'unknown', 170, 7, 999)]
        result = self.run_trace(events)
        self.assertFalse(result['kernel_activity_partition_valid'])
        self.assertEqual(result['kernel_activity_partition_microseconds']['unattributed'], 7)


if __name__ == '__main__':
    unittest.main()
