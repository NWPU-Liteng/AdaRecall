from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pim_backend.execution_route import RouteInput, compile_route, validate_graph, validate_schedule
from pim_backend.lpddr_bank_pim import BankPimConfig
import test_omni_opaque_regions as opaque_fixture


class OpaqueScheduleTests(unittest.TestCase):
    def test_route_validator_rejects_removed_or_timed_region(self):
        original, unused, unused_state, unused_result = opaque_fixture.OpaqueRegionTests().graph()
        for mutation in ('omitted', 'children', 'duration', 'duplicate'):
            graph = deepcopy(original)
            if mutation == 'omitted':
                graph['opaque_regions'] = []
            elif mutation == 'children':
                graph['opaque_regions'][1]['child_event_ids'] = []
            elif mutation == 'duration':
                graph['opaque_regions'][1]['duration_seconds'] = 0
            else:
                graph['opaque_regions'].append(graph['opaque_regions'][1])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_schedule_retains_regions_and_inclusive_unknown_markers(self):
        graph, unused, unused_state, unused_result = opaque_fixture.OpaqueRegionTests().graph()
        graph['linear_events'] = []
        compiled = compile_route([RouteInput('0', 'q', graph)], BankPimConfig())
        schedule = compiled['schedules'][0]
        self.assertEqual(schedule['opaque_regions'], graph['opaque_regions'])
        self.assertEqual(compiled['inclusive_completion_markers'], 2)
        self.assertEqual(compiled['unknown_duration_events'], len(graph['events']))
        completion = graph['opaque_regions'][0]['completion_event']
        schedule['events'][completion]['opaque_region_completion'] = False
        with self.assertRaises(ValueError):
            validate_schedule(graph, schedule['events'], compiled['placement'], compiled['costs'])
