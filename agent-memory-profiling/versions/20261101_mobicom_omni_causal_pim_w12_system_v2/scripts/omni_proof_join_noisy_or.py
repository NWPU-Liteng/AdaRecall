"""SATURATE-style noisy-or evidence selection with exact feasibility checks."""
from __future__ import annotations

import bisect
import math
import time

from omni_proof_join import (augmented_structure, solve_witness_join,
                             _compatible)


def candidate_modality(kind):
    if kind == 'source_text':
        return 'text'
    if kind == 'image_locator':
        return 'image'
    raise ValueError(f'unsupported candidate record kind: {kind}')


def calibrated_probability(calibrator, modality, rerank_score):
    """Interpolate one modality's stored isotonic breakpoints."""
    if modality not in {'text', 'image'}:
        raise ValueError(f'unsupported calibration modality: {modality}')
    if (isinstance(rerank_score, bool)
            or not isinstance(rerank_score, (int, float))
            or not math.isfinite(float(rerank_score))):
        raise ValueError('rerank_score must be finite numeric')
    fitted = calibrator.get('calibration', {}).get(modality)
    if not isinstance(fitted, dict):
        raise ValueError(f'calibrator modality missing: {modality}')
    points = fitted.get('breakpoints', {})
    x = points.get('rerank_score')
    y = points.get('calibrated_probability')
    if (not isinstance(x, list) or not isinstance(y, list)
            or not x or len(x) != len(y)):
        raise ValueError(f'calibrator breakpoints unavailable: {modality}')
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
           for value in x + y):
        raise ValueError(f'calibrator breakpoints must be finite: {modality}')
    if any(float(x[index]) > float(x[index + 1])
           for index in range(len(x) - 1)):
        raise ValueError(f'calibrator breakpoints must be sorted: {modality}')
    score = float(rerank_score)
    if score <= float(x[0]):
        probability = float(y[0])
    elif score >= float(x[-1]):
        probability = float(y[-1])
    else:
        upper = bisect.bisect_right(x, score)
        lower = upper - 1
        x0, x1 = float(x[lower]), float(x[upper])
        y0, y1 = float(y[lower]), float(y[upper])
        probability = (y1 if x1 == x0
                       else y0 + (y1 - y0) * (score - x0) / (x1 - x0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError('calibrated probability must lie in [0, 1]')
    return probability


def witness_effective_probability(witness, calibrator):
    """Heuristic ``min`` of calibrated record probabilities within one witness."""
    records = witness.get('record_scores')
    if not isinstance(records, list) or not records:
        raise ValueError('witness record_scores required for noisy-or')
    probabilities = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('witness record score must be an object')
        probabilities.append(calibrated_probability(
            calibrator, record.get('modality'), record.get('rerank_score')))
    return min(probabilities)


def _validate_inputs(demands, witnesses, relations, levels,
                     max_witnesses_per_demand, fill_redundancy, time_limit):
    if (not isinstance(demands, list) or not demands
            or len({item.get('demand_id') for item in demands}) != len(demands)):
        raise ValueError('nonempty unique demands required')
    demand_ids = [item['demand_id'] for item in demands]
    if any(not isinstance(value, str) or not value for value in demand_ids):
        raise ValueError('demand identities required')
    if (isinstance(levels, bool) or not isinstance(levels, int) or levels < 1
            or isinstance(max_witnesses_per_demand, bool)
            or not isinstance(max_witnesses_per_demand, int)
            or max_witnesses_per_demand < 1
            or not isinstance(fill_redundancy, bool)):
        raise ValueError('positive levels and witness limit required')
    if time_limit is not None and (
            isinstance(time_limit, bool)
            or not isinstance(time_limit, (int, float))
            or not math.isfinite(float(time_limit)) or float(time_limit) <= 0):
        raise ValueError('positive finite time limit required')
    identities = set()
    for witness in witnesses:
        if (not isinstance(witness, dict)
                or witness.get('demand_id') not in set(demand_ids)
                or not isinstance(witness.get('witness_id'), str)
                or not witness['witness_id']
                or witness['witness_id'] in identities
                or not isinstance(witness.get('object_ids'), list)
                or not witness['object_ids']):
            raise ValueError('unique valid witnesses for known demands required')
        identities.add(witness['witness_id'])
    return demand_ids


def _compatible_with_all(candidate, selected, relations):
    return all(_compatible(candidate, {previous['demand_id']: previous},
                           relations)
               for previous in selected)


def _greedy_for_threshold(*, threshold, demands, witnesses, relations, evaluate,
                          probabilities, object_ids_by_witness,
                          max_witnesses_per_demand, cache, started, time_limit,
                          fill_only=False, initial_selected=None,
                          initial_object_ids=frozenset(),
                          initial_assessment=None, initial_coverage=None):
    demand_ids = [demand['demand_id'] for demand in demands]
    selected = list(initial_selected or [])
    selected_ids = {witness['witness_id'] for witness in selected}
    per_demand_count = {
        demand_id: sum(witness['demand_id'] == demand_id
                       for witness in selected)
        for demand_id in demand_ids
    }
    object_ids = frozenset(initial_object_ids)
    assessment = initial_assessment
    if assessment is None:
        assessment = cache.get(object_ids)
        if assessment is None:
            assessment = evaluate(object_ids)
            cache[object_ids] = assessment
    if not assessment.get('feasible'):
        return {'success': False, 'selected': [], 'assessment': assessment,
                'coverage': {demand_id: 0.0 for demand_id in demand_ids},
                'reason': 'empty_union_infeasible', 'states': 0,
                'frontier_peak': 0, 'object_ids': frozenset()}
    current_coverage = ({
        demand_id: float(initial_coverage.get(demand_id, 0.0))
        for demand_id in demand_ids
    } if initial_coverage is not None else
        {demand_id: 0.0 for demand_id in demand_ids})
    target = len(demand_ids) * float(threshold)
    states, frontier_peak = 0, 0

    while True:
        if not fill_only:
            current_objective = sum(
                min(value, float(threshold)) for value in current_coverage.values())
            if current_objective + 1e-12 >= target:
                return {'success': True, 'selected': selected,
                        'assessment': assessment, 'coverage': current_coverage,
                        'reason': None, 'states': states,
                        'frontier_peak': frontier_peak,
                        'object_ids': object_ids}
        if time_limit is not None and time.perf_counter() - started > time_limit:
            return {'success': False, 'selected': selected,
                    'assessment': assessment, 'coverage': current_coverage,
                    'reason': 'time_limit', 'states': states,
                    'frontier_peak': frontier_peak, 'object_ids': object_ids}

        current_objective = sum(
            min(value, float(threshold)) for value in current_coverage.values())
        best = None
        for witness in witnesses:
            witness_id = witness['witness_id']
            demand_id = witness['demand_id']
            if (witness_id in selected_ids
                    or per_demand_count[demand_id] >= max_witnesses_per_demand
                    or not _compatible_with_all(witness, selected, relations)):
                continue
            proposed = object_ids | object_ids_by_witness[witness_id]
            if proposed == object_ids:
                continue
            proposed_assessment = cache.get(proposed)
            if proposed_assessment is None:
                proposed_assessment = evaluate(proposed)
                cache[proposed] = proposed_assessment
            if not proposed_assessment.get('feasible'):
                continue
            probability = probabilities[witness_id]
            new_value = 1.0 - (1.0 - current_coverage[demand_id]) * (1.0 - probability)
            if fill_only:
                gain = new_value - current_coverage[demand_id]
            else:
                gain = ((sum(min(value, float(threshold))
                             for key, value in current_coverage.items()
                             if key != demand_id)
                         + min(new_value, float(threshold)))
                        - current_objective)
            if gain <= 0:
                continue
            old_cost = float(assessment.get('cost', 0.0))
            new_cost = float(proposed_assessment.get('cost', 0.0))
            delta_cost = max(0.0, new_cost - old_cost)
            ratio = gain / max(1.0, delta_cost)
            if (best is None or ratio > best['ratio']
                    or (ratio == best['ratio'] and witness_id < best['witness_id'])):
                best = {'ratio': ratio, 'witness_id': witness_id,
                        'witness': witness, 'assessment': proposed_assessment,
                        'object_ids': proposed, 'new_value': new_value}
        frontier_peak = max(frontier_peak, len(witnesses) - len(selected))
        if best is None:
            success = (fill_only is False
                       and sum(min(value, float(threshold))
                               for value in current_coverage.values()) + 1e-12
                       >= target)
            return {'success': success, 'selected': selected,
                    'assessment': assessment, 'coverage': current_coverage,
                    'reason': None if success else 'no_positive_gain',
                    'states': states, 'frontier_peak': frontier_peak,
                    'object_ids': object_ids}
        selected.append(best['witness'])
        selected_ids.add(best['witness_id'])
        per_demand_count[best['witness']['demand_id']] += 1
        object_ids = best['object_ids']
        assessment = best['assessment']
        current_coverage[best['witness']['demand_id']] = best['new_value']
        states += 1


def _failure_plan(*, demands, witnesses, relations, cache, started,
                  failure_reason):
    demand_ids = [demand['demand_id'] for demand in demands]
    domains = {demand_id: [] for demand_id in demand_ids}
    for witness in witnesses:
        domains[witness['demand_id']].append(witness)
    return {
        'feasible': False,
        'optimal': False,
        'missing_demands': [],
        'states': 0,
        'frontier_peak_states': 0,
        'evaluated_unique_object_unions': len(cache),
        'structure': augmented_structure(domains, relations),
        'support_upper_bound': None,
        'support_gap': None,
        'materialization_cost_gap_upper_bound': None,
        'selected_witnesses': [],
        'accepted_object_ids': [],
        'support': None,
        'cost': None,
        'assessment': None,
        'planning_seconds': time.perf_counter() - started,
        'solver': 'SATURATE-style noisy-or greedy with exact feasibility',
        'boundary': ('heuristic witness probability is the minimum calibrated '
                     'reranker score in a witness; not factual sufficiency or QA'),
        'join_algorithm': 'noisy_or_saturate',
        't_star': None,
        'per_demand_coverage': {demand_id: 0.0 for demand_id in demand_ids},
        'witnesses_per_demand': {demand_id: 0 for demand_id in demand_ids},
        'n_evaluate_calls': len(cache),
        'failure_reason': failure_reason,
    }


def solve_noisy_or_saturate(demands, witnesses, relations, evaluate, *,
                            probability, levels=20,
                            max_witnesses_per_demand=3,
                            fill_redundancy=False, time_limit=2.0):
    """Maximize the worst-demand noisy-or coverage with SATURATE thresholds."""
    demand_ids = _validate_inputs(
        demands, witnesses, relations, levels, max_witnesses_per_demand,
        fill_redundancy, time_limit)
    started = time.perf_counter()
    cache = {}
    object_ids_by_witness = {}
    probabilities = {}
    for witness in witnesses:
        object_ids = frozenset(witness['object_ids'])
        if not object_ids:
            raise ValueError('witness physical objects required')
        object_ids_by_witness[witness['witness_id']] = object_ids
        value = probability(witness)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0):
            raise ValueError('witness probability must lie in [0, 1]')
        probabilities[witness['witness_id']] = float(value)
    if any(not any(witness['demand_id'] == demand_id for witness in witnesses)
           for demand_id in demand_ids):
        return _failure_plan(
            demands=demands, witnesses=witnesses, relations=relations,
            cache=cache, started=started, failure_reason='missing_demand')

    max_probability = max(
        max(probabilities[witness['witness_id']]
            for witness in witnesses if witness['demand_id'] == demand_id)
        for demand_id in demand_ids)
    thresholds = {index / levels for index in range(1, levels + 1)
                  if index / levels <= max_probability + 1e-12}
    thresholds.update(value for value in probabilities.values()
                      if value <= max_probability + 1e-12)
    thresholds = sorted(thresholds)

    best = None
    low, high = 0, len(thresholds) - 1
    while low <= high:
        middle = (low + high) // 2
        threshold = thresholds[middle]
        result = _greedy_for_threshold(
            threshold=threshold, demands=demands, witnesses=witnesses,
            relations=relations, evaluate=evaluate, probabilities=probabilities,
            object_ids_by_witness=object_ids_by_witness,
            max_witnesses_per_demand=max_witnesses_per_demand, cache=cache,
            started=started, time_limit=time_limit)
        if result['success']:
            best = {'threshold': threshold, **result}
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        return _failure_plan(
            demands=demands, witnesses=witnesses, relations=relations,
            cache=cache, started=started,
            failure_reason='no_saturate_threshold_succeeded')

    selected = best['selected']
    coverage = best['coverage']
    if fill_redundancy:
        filler = _greedy_for_threshold(
            threshold=best['threshold'], demands=demands, witnesses=witnesses,
            relations=relations, evaluate=evaluate, probabilities=probabilities,
            object_ids_by_witness=object_ids_by_witness,
            max_witnesses_per_demand=max_witnesses_per_demand, cache=cache,
            started=started, time_limit=time_limit, fill_only=True,
            initial_selected=selected,
            initial_object_ids=best['object_ids'],
            initial_assessment=best['assessment'],
            initial_coverage=coverage)
        if filler['selected']:
            selected = filler['selected']
            coverage = filler['coverage']
            best.update({
                'assessment': filler['assessment'],
                'object_ids': filler['object_ids'],
            })

    per_demand = {
        demand_id: sum(witness['demand_id'] == demand_id
                       for witness in selected)
        for demand_id in demand_ids
    }
    selected_probabilities = [
        probabilities[witness['witness_id']] for witness in selected]
    domains = {demand_id: [] for demand_id in demand_ids}
    for witness in witnesses:
        domains[witness['demand_id']].append(witness)
    return {
        'feasible': True,
        'optimal': False,
        'missing_demands': [],
        'states': best['states'],
        'frontier_peak_states': best['frontier_peak'],
        'evaluated_unique_object_unions': len(cache),
        'structure': augmented_structure(domains, relations),
        'support_upper_bound': None,
        'support_gap': None,
        'materialization_cost_gap_upper_bound': None,
        'selected_witnesses': selected,
        'accepted_object_ids': sorted(best['object_ids']),
        'support': min(selected_probabilities) if selected_probabilities else None,
        'cost': best['assessment']['cost'],
        'assessment': best['assessment'],
        'planning_seconds': time.perf_counter() - started,
        'solver': 'SATURATE-style noisy-or greedy with exact feasibility',
        'boundary': ('heuristic witness probability is the minimum calibrated '
                     'reranker score in a witness; not factual sufficiency or QA'),
        'join_algorithm': 'noisy_or_saturate',
        't_star': best['threshold'],
        'per_demand_coverage': coverage,
        'witnesses_per_demand': per_demand,
        'n_evaluate_calls': len(cache),
        'failure_reason': None,
    }


def solve_configured_witness_join(join_algorithm, demands, witnesses, relations,
                                  evaluate, *, v1_solver=solve_witness_join,
                                  **kwargs):
    """Dispatch the configured solver, retaining the v1 function by name."""
    if join_algorithm == 'dfs_maxmin':
        return v1_solver(demands, witnesses, relations, evaluate, **kwargs)
    if join_algorithm == 'noisy_or_saturate':
        return solve_noisy_or_saturate(
            demands, witnesses, relations, evaluate, **kwargs)
    raise ValueError(f'unknown join algorithm: {join_algorithm}')
