from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch.utils._python_dispatch import _disable_current_modes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_execution_cohort import analyze_graph
from omni_execution_graph import ExecutionGraph
from pim_backend.execution_route import validate_graph


class RecurrentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent_gated_delta_rule = self.hidden_recurrent
        self.causal_conv1d_update = self.hidden_conv

    @staticmethod
    def hidden_recurrent(query, *, initial_state):
        temporary = torch.empty_like(query)
        with _disable_current_modes():
            temporary.copy_(query + initial_state)
            final_state = initial_state + 2
        return temporary, final_state

    @staticmethod
    def hidden_conv(value, conv_state):
        with _disable_current_modes():
            conv_state.add_(1)
            return value + conv_state


class OpaqueRegionTests(unittest.TestCase):
    def graph(self):
        model = RecurrentModule()
        state, query = torch.ones(1, 4), torch.full((1, 4), 2.)
        trace = ExecutionGraph(context_identity='fixed')
        trace.register_external(state, 'cache.layer.state')
        trace.attach_opaque_regions(model)
        with torch.inference_mode(), trace:
            unused = state + 10
            conv = model.causal_conv1d_update(query, state)
            output, final_state = model.recurrent_gated_delta_rule(conv, initial_state=state)
            result = output + final_state
        trace.close()
        return trace.summary(), model, state, result

    def test_hidden_compute_outputs_bind_completion_not_empty_allocation(self):
        graph, unused, state, result = self.graph()
        analyze_graph(graph)
        validate_graph(graph)
        self.assertEqual(result.tolist(), [[10., 10., 10., 10.]])
        self.assertEqual(state.tolist(), [[2., 2., 2., 2.]])
        self.assertEqual(len(graph['opaque_regions']), 2)
        recurrent = graph['opaque_regions'][1]
        self.assertEqual(len(recurrent['child_event_ids']), 1)
        for value in recurrent['output_values']:
            self.assertEqual(graph['values'][value]['producer_event'], recurrent['completion_event'])
            self.assertIn(recurrent['completion_event'], graph['events'][-1]['dependencies'])

    def test_convolution_state_mutation_waits_for_prior_readers(self):
        graph, unused, state, unused_result = self.graph()
        conv = graph['opaque_regions'][0]
        self.assertTrue(conv['mutated_values'])
        completion = graph['events'][conv['completion_event']]
        self.assertIn(0, completion['dependencies'])
        recurrent = graph['events'][graph['opaque_regions'][1]['completion_event']]
        self.assertIn(conv['completion_event'], recurrent['dependencies'])

    def test_regions_preserve_children_without_additive_or_fabricated_timing(self):
        graph, unused, unused_state, unused_result = self.graph()
        for region in graph['opaque_regions']:
            self.assertIsNone(region['duration_seconds'])
            self.assertEqual(region['child_event_ids'], list(range(region['start_event'], region['completion_event'])))

    def test_original_callables_are_restored(self):
        graph, model, unused_state, unused_result = self.graph()
        self.assertEqual(model.causal_conv1d_update, model.hidden_conv)
        self.assertEqual(model.recurrent_gated_delta_rule, model.hidden_recurrent)

    def test_dropped_output_or_child_dependency_fails_audit(self):
        original, unused, unused_state, unused_result = self.graph()
        for mutation in ('output', 'child'):
            graph = deepcopy(original)
            region = graph['opaque_regions'][1]
            if mutation == 'output':
                region['output_values'] = []
            else:
                graph['events'][region['completion_event']]['dependencies'].remove(region['child_event_ids'][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze_graph(graph)
