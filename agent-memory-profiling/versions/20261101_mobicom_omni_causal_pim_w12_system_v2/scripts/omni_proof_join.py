"""Bounded witness join with unique physical-object costs and explicit gaps."""
from __future__ import annotations

import time
import math


def augmented_structure(domains, relations):
    graph = {'d:' + identity: set() for identity in domains}
    for identity, domain in domains.items():
        for witness in domain:
            for member in ['o:' + value for value in witness['object_ids']] + [
                    'b:' + variable for variable in witness.get('bindings', {})]:
                graph.setdefault(member, set()).add('d:' + identity)
                graph['d:' + identity].add(member)
    for relation in relations:
        left, right = 'd:' + relation['left'], 'd:' + relation['right']
        graph[left].add(right)
        graph[right].add(left)
    object_nodes = {node for node in graph if node.startswith('o:')}
    for node in object_nodes:
        graph[node].update(object_nodes - {node})
    width, order = 0, []
    while graph:
        node = min(graph, key=lambda identity: (len(graph[identity]), identity))
        neighbors = graph[node]
        width = max(width, len(neighbors))
        order.append(node)
        for neighbor in neighbors:
            graph[neighbor].update(neighbors - {neighbor})
            graph[neighbor].discard(node)
        del graph[node]
    return {'induced_width_upper_bound': width, 'elimination_order': order,
            'includes_global_exact_render_factor': True, 'optimal_treewidth_claimed': False,
            'separator_dp_used': False,
            'reason': 'nonadditive final processor render is a global resource factor; bounded search retains full object identity'}


def _compatible(candidate, chosen, relations):
    for previous in chosen.values():
        overlap = set(candidate.get('bindings', {})) & set(previous.get('bindings', {}))
        different = {relation['variable'] for relation in relations
            if relation['kind'] == 'different_binding' and {relation['left'], relation['right']}
            == {candidate['demand_id'], previous['demand_id']}}
        if any(candidate['bindings'][key] != previous['bindings'][key] for key in overlap - different):
            return False
    joined = {**chosen, candidate['demand_id']: candidate}
    for relation in relations:
        if relation['left'] not in joined or relation['right'] not in joined:
            continue
        left, right = joined[relation['left']], joined[relation['right']]
        if relation['kind'] == 'source_relation':
            if relation['relation_id'] not in set(left.get('source_relations', [])) | set(right.get('source_relations', [])):
                return False
            continue
        variable = relation['variable']
        if variable not in left.get('bindings', {}) or variable not in right.get('bindings', {}):
            return False
        equal = left['bindings'][variable] == right['bindings'][variable]
        if equal != (relation['kind'] == 'same_binding'):
            return False
    return True


def solve_witness_join(demands, witnesses, relations, evaluate, *, time_limit=2., max_states=100000):
    if (not demands or len({demand['demand_id'] for demand in demands}) != len(demands)
            or not math.isfinite(time_limit) or time_limit <= 0
            or type(max_states) is not int or max_states < 1):
        raise ValueError('nonempty unique demands and finite positive search limits required')
    started = time.perf_counter()
    domains = {demand['demand_id']: [] for demand in demands}
    identities = set()
    for witness in witnesses:
        if (witness['demand_id'] not in domains or not witness['object_ids']
                or witness['witness_id'] in identities or not math.isfinite(float(witness['score']))):
            raise ValueError('unique finite witnesses for known demands with physical objects required')
        identities.add(witness['witness_id'])
        domains[witness['demand_id']].append(witness)
    structure = augmented_structure(domains, relations)
    missing = sorted(identity for identity, values in domains.items() if not values)
    if missing:
        return {'feasible': False, 'optimal': True, 'missing_demands': missing,
                'states': 0, 'planning_seconds': time.perf_counter() - started,
                'support_upper_bound': None, 'support_gap': None, 'selected_witnesses': [],
                'accepted_object_ids': [], 'structure': structure, 'solver': 'bounded witness join'}
    for values in domains.values():
        values.sort(key=lambda item: (-float(item['score']), item['witness_id']))
    order = sorted(domains, key=lambda identity: (len(domains[identity]), identity))
    remaining_upper = [math.inf] * (len(order) + 1)
    for depth in range(len(order) - 1, -1, -1):
        remaining_upper[depth] = min(remaining_upper[depth + 1], float(domains[order[depth]][0]['score']))
    best = None
    pending = [(0, {}, frozenset(), float('inf'))]
    states, frontier_peak = 0, 1
    evaluated = {}
    while pending and states < max_states and time.perf_counter() - started <= time_limit:
        depth, chosen, object_ids, support = pending.pop()
        states += 1
        demand_id = order[depth] if depth < len(order) else None
        if demand_id is None:
            if object_ids not in evaluated:
                evaluated[object_ids] = evaluate(object_ids)
            assessment = evaluated[object_ids]
            if not assessment.get('feasible'):
                continue
            cost = assessment['cost']
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('finite nonnegative materialization cost required')
            candidate = {'support': support, 'cost': cost, 'objects': sorted(object_ids),
                         'witnesses': [chosen[key] for key in order], 'assessment': assessment}
            if best is None or (support, -cost) > (best['support'], -best['cost']):
                best = candidate
            continue
        for witness in reversed(domains[demand_id]):
            if not _compatible(witness, chosen, relations):
                continue
            next_support = min(support, float(witness['score']))
            next_upper = min(next_support, remaining_upper[depth + 1])
            if best is not None and next_upper < best['support']:
                continue
            pending.append((depth + 1, {**chosen, demand_id: witness},
                            object_ids | frozenset(witness['object_ids']), next_support))
        frontier_peak = max(frontier_peak, len(pending))
    optimal = not pending
    upper_candidates = [min(item[3], remaining_upper[item[0]]) for item in pending]
    if best is not None:
        upper_candidates.append(best['support'])
    best_upper = max(upper_candidates, default=None)
    return {'feasible': best is not None, 'optimal': optimal, 'missing_demands': [],
            'states': states, 'frontier_peak_states': frontier_peak,
            'evaluated_unique_object_unions': len(evaluated), 'structure': structure,
            'support_upper_bound': None if best is None else (best['support'] if optimal else best_upper),
            'support_gap': None if best is None else max(0., best_upper - best['support']),
            'materialization_cost_gap_upper_bound': None if best is None else (0 if optimal else best['cost']),
            'selected_witnesses': best['witnesses'] if best else [],
            'accepted_object_ids': best['objects'] if best else [],
            'support': best['support'] if best else None, 'cost': best['cost'] if best else None,
            'assessment': best['assessment'] if best else None,
            'planning_seconds': time.perf_counter() - started,
            'solver': 'bounded witness join with exact unique-object union accounting',
            'boundary': 'optimal only over the retained witness domains; score is a retrieval proxy, not factual sufficiency or QA accuracy'}
