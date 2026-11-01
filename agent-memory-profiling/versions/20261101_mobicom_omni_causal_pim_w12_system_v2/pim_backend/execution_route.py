"""Identity-bound, serial route plans for an explicitly incomplete execution DAG."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .decode_gemv import gemv_cost, select_exact_capacity
from .lpddr_bank_pim import BankPimConfig, BankPimModel


@dataclass(frozen=True)
class RouteInput:
    person_id: str
    question_id: str
    graph: dict


def parameter_identity(labels):
    aliases = sorted(label for label in labels if label.startswith('parameter:'))
    if not aliases:
        raise ValueError('parameter allocation requires a registered parameter label')
    return '||'.join(aliases)


def validate_graph(graph):
    events, values, allocations = graph['events'], graph['values'], graph['allocations']
    if not events or not graph.get('context_identity'):
        raise ValueError('nonempty graph and immutable input identity required')
    for field, records in (('event_id', events), ('value_id', values), ('allocation_id', allocations)):
        if [record[field] for record in records] != list(range(len(records))):
            raise ValueError('contiguous ordered graph identities required')
    for value in values:
        if not 0 <= value['allocation_id'] < len(allocations):
            raise ValueError('unknown storage identity')
        producer = value['producer_event']
        if producer is not None and not 0 <= producer < len(events):
            raise ValueError('unknown producer identity')
    writers, readers = {}, {}
    for event in events:
        event_id, dependencies = event['event_id'], set(event['dependencies'])
        if any(not 0 <= dependency < event_id for dependency in dependencies):
            raise ValueError('future or unknown dependency')
        read = set()
        for value_id in event['input_values']:
            if not 0 <= value_id < len(values):
                raise ValueError('unknown event input')
            value = values[value_id]
            if value['producer_event'] is not None and value['producer_event'] not in dependencies:
                raise ValueError('missing actual input producer')
            read.add(value['allocation_id'])
        for value_id in event['output_values'] + event['mutated_values']:
            if not 0 <= value_id < len(values) or values[value_id]['producer_event'] != event_id:
                raise ValueError('output or mutation producer mismatch')
        written = set(event['write_allocations'])
        if any(not 0 <= allocation < len(allocations) for allocation in written):
            raise ValueError('unknown written allocation')
        if any(values[value]['allocation_id'] not in written for value in event['mutated_values']):
            raise ValueError('mutation must declare its storage write')
        for allocation in read | written:
            if allocation in writers and writers[allocation] not in dependencies:
                raise ValueError('missing read/write-after-write hazard')
        for allocation in written:
            if not readers.get(allocation, set()).issubset(dependencies):
                raise ValueError('missing write-after-read hazard')
            if any(label.startswith('parameter:') for label in allocations[allocation]['external_labels']):
                raise ValueError('weight residency requires immutable parameters')
            writers[allocation], readers[allocation] = event_id, set()
        for allocation in read - written:
            readers.setdefault(allocation, set()).add(event_id)
    regions = graph.get('opaque_regions', [])
    completion_ids = set()
    for region in regions:
        start, end = region['start_event'], region['completion_event']
        if (not region.get('name') or not 0 <= start <= end < len(events)
                or region['end_event'] != end or region['materialized_event'] is not True
                or region['child_event_ids'] != list(range(start, end))
                or region.get('duration_seconds') is not None or end in completion_ids):
            raise ValueError('opaque region must retain a unique untimed inclusive completion')
        completion_ids.add(end)
        event = events[end]
        if (not event.get('opaque_region_completion') or event['operator'] != 'opaque::' + region['name']
                or any(event[field] != region[field] for field in ('input_values', 'output_values', 'mutated_values'))
                or not set(region['child_event_ids']).issubset(event['dependencies'])
                or any(values[value]['producer_event'] != end
                       for value in region['output_values'] + region['mutated_values'])):
            raise ValueError('opaque completion must produce actual outputs after all visible children')
    if completion_ids != {event['event_id'] for event in events if event.get('opaque_region_completion')}:
        raise ValueError('opaque completion event missing its callable region')


def linear_bindings(graph):
    validate_graph(graph)
    bindings = []
    for event in graph['events']:
        scope = event['linear_scope']
        if scope is None or event['operator'] not in {
                'aten.linear.default', 'aten.mm.default', 'aten.addmm.default', 'aten.bmm.default'}:
            continue
        weight_id, activation_id = scope['weight_value'], scope['input_value']
        if any(not 0 <= value < len(graph['values']) for value in (weight_id, activation_id)):
            raise ValueError('unknown Linear scope values')
        weight, activation = graph['values'][weight_id], graph['values'][activation_id]
        allocation_id = weight['allocation_id']
        allocation = graph['allocations'][allocation_id]
        aliases = sorted(label for label in allocation['external_labels'] if label.startswith('parameter:'))
        if 'parameter:' + scope['name'] + '.weight' not in aliases:
            raise ValueError('module name must bind its actual registered parameter')
        inputs, outputs = scope['in_features'], scope['out_features']
        element_bytes = scope['weight_element_bytes']
        if (min(inputs, outputs, element_bytes) <= 0 or weight['shape'] != [outputs, inputs]
                or weight['logical_bytes'] != inputs * outputs * element_bytes):
            raise ValueError('actual weight shape differs from Linear scope')
        eligible = (event['operator'] == 'aten.linear.default' and scope['rows'] == 1
                    and event['input_values'] == [activation_id, weight_id]
                    and activation['shape'][-1] == inputs
                    and math.prod(activation['shape']) == inputs
                    and activation['dtype'] == weight['dtype']
                    and weight['dtype'] in {'torch.bfloat16', 'torch.float16', 'torch.float32'}
                    and weight['stride'] == [inputs, 1] and weight['storage_offset'] == 0
                    and allocation['storage_bytes'] == weight['logical_bytes'])
        bindings.append({'event_id': event['event_id'], **scope,
            'parameter_identity': parameter_identity(aliases), 'parameter_aliases': aliases,
            'weight_allocation': allocation_id, 'weight_dtype': weight['dtype'],
            'eligible': eligible})
    expected = [{'event_id': binding['event_id'], **graph['events'][binding['event_id']]['linear_scope']}
                for binding in bindings]
    if graph['linear_events'] != expected:
        raise ValueError('Linear summary omits, duplicates, reorders or changes actual calls')
    identities = {}
    for binding in bindings:
        identity, allocation = binding['parameter_identity'], binding['weight_allocation']
        if identity in identities and identities[identity] != allocation:
            raise ValueError('one parameter identity cannot identify two physical allocations')
        identities[identity] = allocation
    return bindings


def refresh_bounds(active_cycles, config):
    if active_cycles == 0:
        return {'min_cycles': 0, 'max_cycles': 0}
    available = config.refresh_interval - config.refresh_cycles
    return {'min_cycles': active_cycles + ((active_cycles - 1) // available) * config.refresh_cycles,
            'max_cycles': active_cycles + math.ceil(active_cycles / available) * config.refresh_cycles}


def routed_gemv_cost(binding, config):
    inputs, outputs = binding['in_features'], binding['out_features']
    element_bytes = binding['weight_element_bytes']
    cost = gemv_cost(inputs, outputs, element_bytes, config)
    backend = BankPimModel(config)
    counts = tuple(outputs // config.bank_count + int(bank < outputs % config.bank_count)
                   for bank in range(config.bank_count))
    host_service = max(backend.transfer_cycles(cost['host_external_bytes']),
                       backend._bank_memory(counts, inputs * element_bytes)[0],
                       math.ceil(inputs * outputs / config.host_macs_per_cycle))
    input_bytes = inputs * element_bytes * min(outputs, config.bank_count)
    output_bytes = outputs * 4
    host_phases = [{'phase': 'host_command', 'active_cycles': config.host_command_cycles},
                   {'phase': 'host_streaming_gemv', 'active_cycles': host_service}]
    pim_phases = [
        {'phase': 'tile_commands', 'active_cycles': math.ceil(inputs / 256) * config.host_command_cycles},
        {'phase': 'activation_delivery', 'active_cycles': backend.transfer_cycles(input_bytes),
         'external_bytes': input_bytes},
        {'phase': 'bank_row_service', 'active_cycles': cost['pim_bank_memory_cycles']},
        {'phase': 'bank_mac', 'active_cycles': cost['pim_bank_compute_cycles']},
        {'phase': 'result_return', 'active_cycles': backend.transfer_cycles(output_bytes),
         'external_bytes': output_bytes},
        {'phase': 'completion_fence', 'active_cycles': config.host_fence_cycles}]
    copy_phases = [
        {'phase': 'placement_command', 'active_cycles': config.host_command_cycles},
        {'phase': 'weight_copy', 'active_cycles': backend.transfer_cycles(cost['resident_bytes']),
         'external_bytes': cost['resident_bytes']},
        {'phase': 'placement_fence', 'active_cycles': config.host_fence_cycles}]
    result = {**cost, 'parameter_identity': binding['parameter_identity'], 'phase_costs': {}}
    for route, phases in (('host', host_phases), ('pim', pim_phases), ('placement', copy_phases)):
        active = sum(phase['active_cycles'] for phase in phases)
        elapsed = backend.refresh_elapsed(active)
        result['phase_costs'][route] = {'phases': phases, 'active_cycles': active,
            'phase_zero_elapsed_cycles': elapsed, 'phase_zero_refresh_cycles': elapsed - active,
            **refresh_bounds(active, config)}
    return result


def validate_schedule(graph, schedule, placement, costs):
    bindings = {binding['event_id']: binding for binding in linear_bindings(graph)}
    cache_allocations = {allocation['allocation_id'] for allocation in graph['allocations']
        if any(label.startswith('cache.') for label in allocation['external_labels'])}
    if len(schedule) != len(graph['events']):
        raise ValueError('no observed event may disappear')
    for index, (entry, event) in enumerate(zip(schedule, graph['events'])):
        for field in ('event_id', 'operator', 'dependencies', 'input_values', 'output_values',
                      'mutated_values', 'write_allocations'):
            if entry[field] != event[field]:
                raise ValueError('schedule changes actual execution order or dependencies')
        if entry.get('opaque_region_completion', False) != event.get('opaque_region_completion', False):
            raise ValueError('inclusive completion cannot be relabeled as another hardware kernel')
        expected_waits = sorted(set(event['dependencies']) | ({index - 1} if index else set()))
        if entry['wait_for_events'] != expected_waits or entry['absolute_start_cycle'] is not None:
            raise ValueError('serial completion barriers and unknown absolute time required')
        if entry['state_writeback_allocations'] != sorted(set(event['write_allocations']) & cache_allocations):
            raise ValueError('state writeback barriers cannot disappear')
        binding = bindings.get(index)
        if binding is not None and binding['eligible']:
            identity = binding['parameter_identity']
            route = 'pim' if identity in placement else 'host'
            if (entry['route'] != route or entry['parameter_identity'] != identity
                    or entry['cost_identity'] != identity or entry['kind'] != 'modeled_linear'
                    or entry['placement_dependency'] != (identity if route == 'pim' else None)
                    or entry['modeled_cycle_range'] != {
                        field: costs[identity]['phase_costs'][route][field]
                        for field in ('min_cycles', 'max_cycles')}):
                raise ValueError('route must bind actual weight identity, placement and cost')
        elif (entry['route'] != 'unmodeled' or entry['modeled_cycle_range'] is not None):
            raise ValueError('opaque or unsupported computation must remain unknown')


def compile_route(requests: list[RouteInput], config: BankPimConfig, *, reserved_bytes=0):
    if not requests or len({(item.person_id, item.question_id) for item in requests}) != len(requests):
        raise ValueError('nonempty, unique fixed request stream required')
    if not isinstance(reserved_bytes, int) or not 0 <= reserved_bytes <= config.pim_capacity_bytes:
        raise ValueError('known shared-pool reservations must fit before any weight placement')
    first = linear_bindings(requests[0].graph)
    costs = {}
    for binding in first:
        if binding['eligible']:
            identity = binding['parameter_identity']
            cost = routed_gemv_cost(binding, config)
            if identity in costs and costs[identity] != cost:
                raise ValueError('one physical weight requires one consistent cost')
            costs[identity] = cost
    placement_candidates = {identity: {**cost,
        'host_cycles': cost['phase_costs']['host']['min_cycles'],
        'pim_cycles': cost['phase_costs']['pim']['max_cycles']}
        for identity, cost in costs.items()}
    selected, resident_bytes, gain = select_exact_capacity(placement_candidates, config.pim_capacity_bytes - reserved_bytes)
    signature_fields = ('parameter_identity', 'in_features', 'out_features', 'rows',
                        'weight_element_bytes', 'weight_dtype', 'eligible')
    signature = [tuple(binding[field] for field in signature_fields) for binding in first]
    schedules = []
    for request in requests:
        bindings = linear_bindings(request.graph)
        if [tuple(binding[field] for field in signature_fields) for binding in bindings] != signature:
            raise ValueError('later requests cannot change a first-step-frozen placement program')
        cache_allocations = {allocation['allocation_id'] for allocation in request.graph['allocations']
            if any(label.startswith('cache.') for label in allocation['external_labels'])}
        by_event = {binding['event_id']: binding for binding in bindings}
        schedule = []
        totals = {name: 0 for name in ('host_min_cycles', 'host_max_cycles', 'hybrid_min_cycles',
                   'hybrid_max_cycles', 'host_external_bytes', 'hybrid_external_bytes', 'pim_calls')}
        for event in request.graph['events']:
            event_id, operator = event['event_id'], event['operator']
            binding = by_event.get(event_id)
            entry = {field: event[field] for field in ('event_id', 'operator', 'dependencies',
                     'input_values', 'output_values', 'mutated_values', 'write_allocations')}
            if event.get('opaque_region_completion'):
                entry['opaque_region_completion'] = True
            entry.update({'wait_for_events': sorted(set(event['dependencies']) | (
                {event_id - 1} if event_id else set())), 'absolute_start_cycle': None,
                'state_writeback_allocations': sorted(set(event['write_allocations']) & cache_allocations),
                'route': 'unmodeled', 'modeled_cycle_range': None,
                'kind': 'opaque_kernel' if not operator.startswith('aten.') or
                        'scaled_dot_product_attention' in operator else 'unmodeled_operator'})
            if binding is not None and binding['eligible']:
                identity = binding['parameter_identity']
                route = 'pim' if identity in selected else 'host'
                cost = costs[identity]
                entry.update({'route': route, 'kind': 'modeled_linear', 'parameter_identity': identity,
                    'weight_allocation': binding['weight_allocation'], 'cost_identity': identity,
                    'placement_dependency': identity if route == 'pim' else None,
                    'modeled_cycle_range': {field: cost['phase_costs'][route][field]
                                           for field in ('min_cycles', 'max_cycles')}})
                for field in ('min_cycles', 'max_cycles'):
                    totals['host_' + field] += cost['phase_costs']['host'][field]
                    totals['hybrid_' + field] += cost['phase_costs'][route][field]
                totals['host_external_bytes'] += cost['host_external_bytes']
                totals['hybrid_external_bytes'] += cost[route + '_external_bytes']
                totals['pim_calls'] += route == 'pim'
            schedule.append(entry)
        validate_schedule(request.graph, schedule, selected, costs)
        schedules.append({'person_id': request.person_id, 'question_id': request.question_id,
            'context_identity': request.graph['context_identity'], 'events': schedule,
            'opaque_regions': request.graph.get('opaque_regions', []),
            'known_linear_subgraph': totals})
    placement = {identity: {'resident_bytes': costs[identity]['resident_bytes'],
        'copy_external_bytes': costs[identity]['resident_bytes'],
        'phase_cost_identity': identity, 'ready_before_first_request': True} for identity in sorted(selected)}
    totals = {field: sum(item['known_linear_subgraph'][field] for item in schedules)
              for field in schedules[0]['known_linear_subgraph']}
    copy_min = sum(costs[identity]['phase_costs']['placement']['min_cycles'] for identity in selected)
    copy_max = sum(costs[identity]['phase_costs']['placement']['max_cycles'] for identity in selected)
    per_step_gain = totals['host_min_cycles'] // len(requests) - totals['hybrid_max_cycles'] // len(requests)
    return {'schema_version': 'omni-execution-driven-pim-route-v1', 'requests': len(requests),
        'observed_events': sum(len(item['events']) for item in schedules),
        'modeled_linear_calls': sum(entry['kind'] == 'modeled_linear' for item in schedules for entry in item['events']),
        'opaque_events': sum(entry['kind'] == 'opaque_kernel' for item in schedules for entry in item['events']),
        'unknown_duration_events': sum(entry['route'] == 'unmodeled' for item in schedules for entry in item['events']),
        'inclusive_completion_markers': sum(len(item['opaque_regions']) for item in schedules),
        'state_writeback_barriers': sum(bool(entry['state_writeback_allocations']) for item in schedules for entry in item['events']),
        'candidate_matrices': len(costs), 'resident_matrices': len(selected), 'resident_bytes': resident_bytes,
        'capacity_bytes': config.pim_capacity_bytes, 'robust_per_unique_matrix_cycle_gain': gain,
        'shared_reserved_bytes': reserved_bytes,
        'weight_placement_budget_bytes': config.pim_capacity_bytes - reserved_bytes,
        'known_joint_pool_bytes': resident_bytes + reserved_bytes,
        'placement_cycle_range': {'min_cycles': copy_min, 'max_cycles': copy_max},
        'known_linear_subgraph': totals,
        'warm_known_linear_speedup_lower_scenario': totals['host_min_cycles'] / totals['hybrid_max_cycles'] if totals['hybrid_max_cycles'] else None,
        'cold_first_step_known_linear_speedup_lower_scenario': totals['host_min_cycles'] / (
            totals['hybrid_max_cycles'] + len(requests) * copy_max) if totals['hybrid_max_cycles'] else None,
        'hypothetical_break_even_decode_steps': math.ceil(copy_max / per_step_gain) if per_step_gain > 0 else None,
        'placement': placement, 'costs': costs, 'schedules': schedules,
        'placement_policy': 'exact integer capacity DP over unique physical weights in first observed decode step; maximize host best-refresh minus PIM worst-refresh per one invocation; no future calls, labels, output lengths or amortization oracle',
        'schedule_policy': 'observed serial order, all data/alias edges, explicit completion barriers; no overlap inferred through unknown kernels',
        'refresh_policy': 'phase-independent primitive elapsed bounds; no unknown kernel assigned a refresh phase or zero duration',
        'phase_model_boundary': 'serialized aggregate tile phases, not a microcommand trace; host GEMV uses a compute/memory roofline; layout construction and opaque work remain unmodeled',
        'unobserved_regions': ['opaque kernel internals, hardware traffic and elapsed costs remain unknown; callable completions are inclusive and cannot be added to child event costs'],
        'capacity_scope': 'fixed PIM pool, known nonweight reservations deducted before weight placement; not full-device capacity qualification or a 12GB workload scenario',
        'actual_pim_routing_executed': False, 'new_quality_evaluated': False,
        'end_to_end_speedup': None, 'publication_ready': False,
        'boundary': 'known-linear-subgraph architecture-assumption scenarios only; all other visible work and unobserved regions remain unknown; no real-chip, measured-latency or E2E acceleration claim'}
