from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_execution_graph import ExecutionGraph


class ExecutionGraphTests(unittest.TestCase):
    def test_observer_keeps_values_and_data_dependency(self):
        value = torch.tensor([2., 3.])
        trace = ExecutionGraph(context_identity='fixed')
        trace.register_external(value, 'activation')
        with trace:
            intermediate = value * 2
            result = intermediate + 1
        self.assertTrue(torch.equal(result, torch.tensor([5., 7.])))
        graph = trace.summary()
        self.assertEqual(graph['events'][1]['dependencies'], [0])
        self.assertEqual(graph['allocations'][0]['external_labels'], ['activation'])

    def test_alias_write_waits_for_read_and_future_read_waits_for_write(self):
        value = torch.tensor([2., 3.])
        trace = ExecutionGraph(context_identity='fixed')
        with torch.inference_mode(), trace:
            alias = value.view(2)
            unused = alias + 1
            value.add_(2)
            result = alias * 3
        events = trace.summary()['events']
        self.assertIn(1, events[2]['dependencies'])
        self.assertIn(2, events[3]['dependencies'])
        self.assertTrue(torch.equal(result, torch.tensor([12., 15.])))
        self.assertEqual(unused.tolist(), [3., 4.])

    def test_shared_linear_input_is_actual_identity_not_name_matching(self):
        class Branches(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(3, 4, bias=False)
                self.second = torch.nn.Linear(3, 4, bias=False)

            def forward(self, value):
                return self.first(value) + self.second(value)

        model = Branches().eval()
        value = torch.ones(1, 3)
        expected = model(value)
        trace = ExecutionGraph(context_identity='fixed')
        trace.attach_linears(model)
        with torch.inference_mode(), trace:
            result = model(value)
        trace.close()
        self.assertTrue(torch.equal(expected, result))
        group = trace.summary()['shared_linear_input_values']
        self.assertEqual(len(group), 1)
        self.assertEqual(len(group[0]['linear_events']), 2)

    def test_event_limit_fails_instead_of_truncating(self):
        value = torch.ones(2)
        trace = ExecutionGraph(context_identity='fixed', max_events=1)
        with self.assertRaises(ValueError), trace:
            intermediate = value + 1
            intermediate + 2

    def test_inplace_versions_are_distinct_without_tensor_version_counter(self):
        trace = ExecutionGraph(context_identity='fixed')
        with torch.inference_mode(), trace:
            value = torch.ones(2)
            value.add_(1)
            result = value * 2
        graph = trace.summary()
        mutated = graph['events'][1]['mutated_values'][0]
        self.assertIn(mutated, graph['events'][2]['input_values'])
        self.assertEqual(result.tolist(), [4., 4.])

    def test_invisible_opaque_callable_gets_a_materialized_boundary(self):
        class Opaque(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.recurrent_gated_delta_rule = lambda value: value

            def forward(self, value):
                return self.recurrent_gated_delta_rule(value)

        model = Opaque()
        trace = ExecutionGraph(context_identity='fixed')
        trace.attach_opaque_regions(model)
        value = torch.ones(2)
        with trace:
            output = model(value)
        trace.close()
        graph = trace.summary()
        self.assertTrue(torch.equal(output, torch.ones(2)))
        self.assertEqual(len(graph['opaque_regions']), 1)
        self.assertTrue(graph['opaque_regions'][0]['materialized_event'])
        self.assertEqual(len(graph['events']), 1)

    def test_visible_nested_work_is_not_double_counted(self):
        class Visible(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.recurrent_gated_delta_rule = lambda value: value + 1

            def forward(self, value):
                return self.recurrent_gated_delta_rule(value)

        model = Visible()
        trace = ExecutionGraph(context_identity='fixed')
        trace.attach_opaque_regions(model)
        value = torch.ones(2)
        with trace:
            model(value)
        trace.close()
        graph = trace.summary()
        region = graph['opaque_regions'][0]
        self.assertTrue(region['materialized_event'])
        self.assertEqual(len(graph['events']), 2)
        self.assertEqual(region['child_event_ids'], [0])
        self.assertIsNone(region['duration_seconds'])
        self.assertEqual(graph['values'][region['output_values'][0]]['producer_event'], 1)


if __name__ == '__main__':
    unittest.main()
