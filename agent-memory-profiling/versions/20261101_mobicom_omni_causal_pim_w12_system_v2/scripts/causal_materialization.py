"""Bounded exact search on a fixed request of immutable closed memory units."""
from __future__ import annotations

import math
from numbers import Integral
import time


def solve_materialization_plan(bundles, costs, *, budget, max_nodes=10000, evidence_scores=None):
    started = time.perf_counter()
    if (not bundles or not costs or not math.isfinite(budget) or budget < 1
            or not isinstance(max_nodes, Integral) or max_nodes < 1
            or any(not math.isfinite(cost) or cost <= 0 for cost in costs)):
        raise ValueError('invalid materialization input')
    weighted = evidence_scores is not None
    evidence_scores = list(evidence_scores) if weighted else [1.0] * len(bundles)
    if (len(evidence_scores) != len(bundles)
            or any(not math.isfinite(score) or not 0 <= score <= 1 for score in evidence_scores)):
        raise ValueError('one finite frozen evidence score in [0, 1] required per seed')
    ratios = [float(score).as_integer_ratio() for score in evidence_scores]
    denominator = math.lcm(*(ratio[1] for ratio in ratios))
    input_units = [numerator * (denominator // divisor) for numerator, divisor in ratios]
    normalized, representative_ranks, input_to_unique = [], [], []
    unique = {}
    for rank, bundle in enumerate(bundles):
        frozen = frozenset(bundle)
        if not frozen or any(not isinstance(index, Integral) or index < 0 or index >= len(costs)
                             for index in frozen):
            raise ValueError('invalid closed anchor')
        if frozen not in unique:
            unique[frozen] = len(normalized)
            normalized.append(frozen)
            representative_ranks.append(rank)
        input_to_unique.append(unique[frozen])
    utility_units = [0] * len(normalized)
    for rank, closure in enumerate(input_to_unique):
        utility_units[closure] = max(utility_units[closure], input_units[rank])
    infeasible = [index for index, bundle in enumerate(normalized)
                  if sum(costs[item] for item in bundle) > budget]
    feasible = [index for index in range(len(normalized)) if index not in infeasible]
    remaining_units = [0] * (len(feasible) + 1)
    for position in range(len(feasible) - 1, -1, -1):
        remaining_units[position] = remaining_units[position + 1] + utility_units[feasible[position]]
    best_union, best_accepted, best_cost = frozenset(), (), 0
    best_score = (0, 0, ())
    pending = [(0, frozenset(), (), 0)]
    nodes = 0
    while pending and nodes < max_nodes:
        position, union, accepted, used = pending.pop()
        nodes += 1
        accepted_units = sum(utility_units[index] for index in accepted)
        score = (accepted_units, -used, tuple(-representative_ranks[index] for index in accepted))
        if score > best_score:
            best_score = score
            best_union, best_accepted, best_cost = union, accepted, used
        if position == len(feasible) or accepted_units + remaining_units[position] < best_score[0]:
            continue
        closure_index = feasible[position]
        proposal = union | normalized[closure_index]
        proposal_cost = used + sum(costs[index] for index in normalized[closure_index] - union)
        pending.append((position + 1, union, accepted, used))
        if proposal_cost <= budget:
            pending.append((position + 1, proposal, accepted + (closure_index,), proposal_cost))
    upper_units = max([best_score[0]] + [sum(utility_units[index] for index in accepted)
                                        + remaining_units[position]
                                        for position, _, accepted, _ in pending])
    references = {index: sum(index in normalized[rank] for rank in best_accepted)
                  for index in best_union}
    independent_tokens = sum(costs[index] for rank in best_accepted for index in normalized[rank])
    accepted_inputs = [rank for rank, closure in enumerate(input_to_unique) if closure in best_accepted]
    accepted_input_tokens = sum(costs[index] for rank in accepted_inputs for index in bundles[rank])
    return {
        'materialized': sorted(best_union), 'used_tokens': best_cost,
        'bundle_count': len(normalized), 'input_bundle_count': len(bundles),
        'deduplicated_bundle_count': len(bundles) - len(normalized),
        'input_to_unique_closure': input_to_unique,
        'unique_closures': [sorted(bundle) for bundle in normalized],
        'accepted_closure_indices': list(best_accepted),
        'infeasible_closure_indices': infeasible,
        'accepted_anchor_ranks': accepted_inputs,
        'rejected_anchor_ranks': [rank for rank in range(len(bundles)) if rank not in accepted_inputs],
        'anchor_satisfaction': [int(rank in accepted_inputs) for rank in range(len(bundles))],
        'objective': best_score[0] / denominator, 'upper_bound': upper_units / denominator,
        'accepted_closure_count': len(best_accepted),
        'objective_schema': ('maximum frozen neural evidence score over unique closures, then minimum union cost'
                             if weighted else 'maximum feasible closed-anchor coverage, then minimum shared materialization cost'),
        'evidence_score_source': 'frozen reranker yes probability; not calibrated answer probability' if weighted else 'unit coverage',
        'input_evidence_scores': evidence_scores,
        'utility_denominator': denominator, 'closure_utility_units': utility_units,
        'objective_units': best_score[0], 'upper_bound_units': upper_units,
        'absolute_gap_units': upper_units - best_score[0],
        'duplicate_score_aggregation': 'maximum seed score; no additive reward for duplicate closures',
        'coverage_unit': 'unique observable closure; duplicate seeds do not increase utility',
        'tie_break': 'earliest frozen reranker rank after equal coverage and cost',
        'optimality_scope': 'supplied deduplicated closure set only; no answer-quality guarantee',
        'optimality_certified': not pending,
        'shared_references': {str(index): references[index] for index in sorted(best_union)},
        'independent_tokens': independent_tokens,
        'avoided_duplicate_tokens': independent_tokens - best_cost,
        'duplicate_seed_reference_tokens_avoided': accepted_input_tokens - independent_tokens,
        'expanded_nodes': nodes, 'search_exhausted': not pending,
        'absolute_gap': (upper_units - best_score[0]) / denominator,
        'secondary_objective_certified': not pending,
        'seconds': time.perf_counter() - started,
        'boundary': 'structural closure, not factual causality or semantic completeness; over-budget closures rejected explicitly',
    }
