from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.execution_route import (RouteInput, compile_route, linear_bindings,
    refresh_bounds, validate_schedule)
from pim_backend.lpddr_bank_pim import BankPimConfig, BankPimModel


def graph_fixture(*, inputs=2048, outputs=2048, aliases=None):
    scope = {'name': 'projection', 'input_value': 0, 'weight_value': 1,
        'in_features': inputs, 'out_features': outputs, 'rows': 1, 'weight_element_bytes': 2}
    def value(value_id, allocation, producer, shape):
        return {'value_id': value_id, 'allocation_id': allocation, 'producer_event': producer,
            'shape': shape, 'stride': [shape[-1], 1], 'storage_offset': 0,
            'dtype': 'torch.bfloat16', 'logical_bytes': shape[0] * shape[1] * 2}
    values = [value(0, 0, None, [1, inputs]), value(1, 1, None, [outputs, inputs]),
              value(2, 2, 0, [1, outputs]), value(3, 2, 1, [1, outputs])]
    events = [
        {'event_id': 0, 'operator': 'aten.linear.default', 'input_values': [0, 1],
         'output_values': [2], 'mutated_values': [], 'dependencies': [], 'write_allocations': [2],
         'linear_scope': scope, 'nondeterministic_seeded': False},
        {'event_id': 1, 'operator': 'DaoAILab._causal_conv1d_update_cpp.default',
         'input_values': [2], 'output_values': [3], 'mutated_values': [3], 'dependencies': [0],
         'write_allocations': [2], 'linear_scope': None, 'nondeterministic_seeded': False}]
    allocations = [{'allocation_id': 0, 'storage_bytes': inputs * 2, 'external_labels': []},
        {'allocation_id': 1, 'storage_bytes': inputs * outputs * 2,
         'external_labels': aliases or ['parameter:projection.weight']},
        {'allocation_id': 2, 'storage_bytes': outputs * 2, 'external_labels': ['cache.layer.state']}]
    return {'context_identity': 'immutable', 'events': events, 'values': values,
        'allocations': allocations, 'linear_events': [{'event_id': 0, **scope}]}


class ExecutionRouteTests(unittest.TestCase):
    def compile(self, graph=None, config=None):
        return compile_route([RouteInput('0', 'q', graph or graph_fixture())], config or BankPimConfig())

    def test_actual_identity_route_and_unknown_kernel_are_retained(self):
        result = self.compile()
        self.assertEqual(result['modeled_linear_calls'], 1)
        self.assertEqual(result['opaque_events'], 1)
        self.assertEqual(result['resident_matrices'], 1)
        events = result['schedules'][0]['events']
        self.assertEqual(events[0]['parameter_identity'], 'parameter:projection.weight')
        self.assertIsNone(events[1]['modeled_cycle_range'])
        self.assertIsNone(events[0]['absolute_start_cycle'])
        self.assertIsNone(result['end_to_end_speedup'])
        self.assertFalse(result['actual_pim_routing_executed'])

    def test_no_event_can_disappear_or_change_order(self):
        graph = graph_fixture()
        result = self.compile(graph)
        for mutation in ('omit', 'swap', 'operator'):
            schedule = deepcopy(result['schedules'][0]['events'])
            if mutation == 'omit':
                schedule.pop()
            elif mutation == 'swap':
                schedule.reverse()
            else:
                schedule[1]['operator'] = 'aten.add.Tensor'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_schedule(graph, schedule, result['placement'], result['costs'])

    def test_missing_state_writeback_hazard_is_rejected(self):
        graph = graph_fixture()
        graph['events'][1]['dependencies'] = []
        with self.assertRaises(ValueError):
            self.compile(graph)

    def test_state_writebacks_force_serial_completion_barriers(self):
        events = self.compile()['schedules'][0]['events']
        self.assertEqual(events[1]['state_writeback_allocations'], [2])
        self.assertEqual(events[1]['wait_for_events'], [0])

    def test_state_barrier_cannot_be_removed_from_exported_schedule(self):
        graph = graph_fixture()
        result = self.compile(graph)
        schedule = deepcopy(result['schedules'][0]['events'])
        schedule[1]['state_writeback_allocations'] = []
        with self.assertRaises(ValueError):
            validate_schedule(graph, schedule, result['placement'], result['costs'])

    def test_shared_reservations_precede_exact_placement(self):
        config = BankPimConfig(pim_capacity_bytes=9 * 1024**2)
        requests = [RouteInput('0', 'q', graph_fixture())]
        result = compile_route(requests, config, reserved_bytes=2 * 1024**2)
        self.assertEqual(result['resident_matrices'], 0)
        self.assertLessEqual(result['known_joint_pool_bytes'], config.pim_capacity_bytes)
        with self.assertRaises(ValueError):
            compile_route(requests, config, reserved_bytes=config.pim_capacity_bytes + 1)

    def test_parameter_scope_cannot_claim_another_weight(self):
        graph = graph_fixture()
        graph['allocations'][1]['external_labels'] = ['parameter:other.weight']
        with self.assertRaises(ValueError):
            self.compile(graph)

    def test_route_parameter_or_cost_identity_cannot_be_swapped(self):
        graph = graph_fixture()
        result = self.compile(graph)
        for field in ('parameter_identity', 'cost_identity'):
            schedule = deepcopy(result['schedules'][0]['events'])
            schedule[0][field] = 'parameter:other.weight'
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_schedule(graph, schedule, result['placement'], result['costs'])

    def test_summary_cannot_omit_or_change_a_linear_call(self):
        for mutation in ('omit', 'shape'):
            graph = graph_fixture()
            if mutation == 'omit':
                graph['linear_events'] = []
            else:
                graph['linear_events'][0]['rows'] = 5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.compile(graph)

    def test_tied_parameter_aliases_are_one_physical_candidate(self):
        aliases = ['parameter:embed_tokens.weight', 'parameter:projection.weight']
        result = self.compile(graph_fixture(aliases=aliases))
        self.assertEqual(result['candidate_matrices'], 1)
        self.assertIn('||'.join(sorted(aliases)), result['placement'])

    def test_later_request_or_output_metadata_cannot_change_placement(self):
        graph = graph_fixture()
        first = self.compile(graph)
        later = deepcopy(graph)
        later['answer_token_ids'] = list(range(1000))
        later['observed_future_decode_calls'] = 1000
        combined = compile_route([RouteInput('0', 'q', graph), RouteInput('1', 'r', later)], BankPimConfig())
        self.assertEqual(first['placement'], combined['placement'])
        self.assertEqual(first['robust_per_unique_matrix_cycle_gain'], combined['robust_per_unique_matrix_cycle_gain'])

    def test_later_new_shapes_require_new_program_not_oracle_placement(self):
        with self.assertRaises(ValueError):
            compile_route([RouteInput('0', 'q', graph_fixture()),
                RouteInput('1', 'r', graph_fixture(outputs=4096))], BankPimConfig())

    def test_capacity_buffer_and_unprofitable_ops_fall_back_to_host(self):
        for config, graph in ((BankPimConfig(pim_capacity_bytes=1), graph_fixture()),
                (BankPimConfig(local_buffer_bytes_per_bank=1), graph_fixture()),
                (BankPimConfig(), graph_fixture(inputs=1, outputs=1))):
            result = self.compile(graph, config)
            self.assertEqual(result['resident_matrices'], 0)
            self.assertEqual(result['schedules'][0]['events'][0]['route'], 'host')

    def test_copy_commands_fences_refresh_and_transfers_are_charged(self):
        result = self.compile()
        cost = result['costs']['parameter:projection.weight']
        for route in ('host', 'pim', 'placement'):
            phases = cost['phase_costs'][route]
            self.assertEqual(sum(phase['active_cycles'] for phase in phases['phases']), phases['active_cycles'])
            self.assertGreater(phases['phase_zero_refresh_cycles'], 0)
        self.assertGreater(cost['phase_costs']['placement']['active_cycles'],
                           BankPimModel(BankPimConfig()).transfer_cycles(result['resident_bytes']))
        self.assertLess(result['cold_first_step_known_linear_speedup_lower_scenario'],
                        result['warm_known_linear_speedup_lower_scenario'])

    def test_refresh_interval_bounds_cover_every_possible_start_phase(self):
        config = BankPimConfig(refresh_interval=31, refresh_cycles=7)
        backend = BankPimModel(config)
        for active in (0, 1, 23, 24, 25, 48, 91):
            elapsed = [backend.refresh_elapsed(active, start) for start in range(config.refresh_interval)]
            bounds = refresh_bounds(active, config)
            self.assertEqual(bounds['min_cycles'], min(elapsed))
            self.assertEqual(bounds['max_cycles'], max(elapsed))

    def test_parameter_mutations_reject_residency(self):
        graph = graph_fixture()
        graph['events'][1]['write_allocations'].append(1)
        with self.assertRaises(ValueError):
            self.compile(graph)

    def test_unsupported_bias_or_layout_is_unknown_not_free(self):
        graph = graph_fixture()
        graph['values'][1]['stride'] = [1, 2048]
        result = self.compile(graph)
        self.assertEqual(result['modeled_linear_calls'], 0)
        self.assertIsNone(result['schedules'][0]['events'][0]['modeled_cycle_range'])


if __name__ == '__main__':
    unittest.main()
