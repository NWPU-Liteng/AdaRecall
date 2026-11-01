from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_execution_cohort import analyze_graph
from omni_execution_graph import ExecutionGraph


class ExecutionAuditTests(unittest.TestCase):
    def graph(self):
        value = torch.ones(1, 3)
        trace = ExecutionGraph(context_identity='fixed')
        trace.register_external(value, 'cache.layer_0.state')
        with torch.inference_mode(), trace:
            copy = value * 2
            value.add_(copy)
            value * 3
        return trace.summary()

    def test_visible_state_mutation_and_topology(self):
        summary = analyze_graph(self.graph())
        self.assertEqual(summary['events'], 3)
        self.assertEqual(summary['registered_cache_storage_bytes'], 12)
        self.assertEqual(summary['visible_mutated_cache_storage_bytes'], 12)

    def test_missing_hazards_or_input_producer_are_rejected(self):
        graph = self.graph()
        graph['events'][2]['dependencies'] = []
        with self.assertRaises(ValueError):
            analyze_graph(graph)

    def test_future_or_invalid_allocation_is_rejected(self):
        for mutation in ('future', 'storage'):
            graph = self.graph()
            if mutation == 'future':
                graph['events'][0]['dependencies'] = [1]
            else:
                graph['values'][0]['allocation_id'] = 999
            with self.assertRaises(ValueError):
                analyze_graph(graph)

    def test_missing_write_after_read_hazard_is_rejected(self):
        trace = ExecutionGraph(context_identity='fixed')
        value = torch.ones(2)
        with torch.inference_mode(), trace:
            unused = value + 1
            value.add_(1)
        graph = trace.summary()
        graph['events'][1]['dependencies'] = []
        self.assertEqual(unused.tolist(), [2., 2.])
        with self.assertRaises(ValueError):
            analyze_graph(graph)

    def test_mutation_through_alias_does_not_create_false_shared_input(self):
        class Branches(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(3, 4, bias=False)
                self.second = torch.nn.Linear(3, 4, bias=False)

            def forward(self, value):
                alias = value.view(1, 3)
                first = self.first(value)
                alias.add_(1)
                return first + self.second(value)

        model = Branches().eval()
        trace = ExecutionGraph(context_identity='fixed')
        for name, parameter in model.named_parameters():
            trace.register_external(parameter, 'parameter:' + name)
        trace.attach_linears(model)
        with torch.inference_mode(), trace:
            model(torch.ones(1, 3))
        trace.close()
        summary = analyze_graph(trace.summary())
        self.assertEqual(summary['observed_linear_calls'], 2)
        self.assertEqual(summary['shared_linear_input_groups_with_unchanged_write_epoch'], [])


if __name__ == '__main__':
    unittest.main()
