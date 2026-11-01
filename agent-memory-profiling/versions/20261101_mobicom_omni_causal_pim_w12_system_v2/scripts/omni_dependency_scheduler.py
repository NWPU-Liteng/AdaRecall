"""Exact small-instance optimizer for dependency-closed materialization states.

This is an algorithmic validation primitive, not a serving heuristic. It refuses
to approximate large instances until a separately proven solver is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterializationState:
    state_id: str
    resident_bytes: int
    net_benefit: int
    dependencies: tuple[str, ...] = ()


def solve_exact(states, capacity_bytes):
    states = tuple(states)
    if len(states) > 20:
        raise ValueError('enumeration oracle limited to 20 states; not a scalable serving solver')
    if capacity_bytes < 0 or len({state.state_id for state in states}) != len(states):
        raise ValueError('unique states and nonnegative capacity required')
    by_id = {state.state_id: state for state in states}
    if any(state.resident_bytes < 0 for state in states):
        raise ValueError('state residency must be nonnegative')
    for state in states:
        if any(dependency not in by_id for dependency in state.dependencies):
            raise ValueError('dependency closure references an unknown state')
    visiting, visited = set(), set()

    def visit(state_id):
        if state_id in visiting:
            raise ValueError('materialization graph must be acyclic')
        if state_id in visited:
            return
        visiting.add(state_id)
        for dependency in by_id[state_id].dependencies:
            visit(dependency)
        visiting.remove(state_id)
        visited.add(state_id)

    for state in states:
        visit(state.state_id)

    best = (0, 0, ())
    for mask in range(1 << len(states)):
        selected = {states[index].state_id for index in range(len(states)) if mask & (1 << index)}
        closure = set(selected)
        changed = True
        while changed:
            changed = False
            for state_id in tuple(closure):
                for dependency in by_id[state_id].dependencies:
                    if dependency not in closure:
                        closure.add(dependency)
                        changed = True
        if closure != selected:
            continue
        resident = sum(by_id[state_id].resident_bytes for state_id in closure)
        if resident > capacity_bytes:
            continue
        benefit = sum(by_id[state_id].net_benefit for state_id in closure)
        candidate = (benefit, -resident, tuple(sorted(closure)))
        if candidate > best:
            best = candidate
    return {
        'selected_state_ids': list(best[2]),
        'net_benefit': best[0],
        'resident_bytes': -best[1],
        'capacity_bytes': capacity_bytes,
        'optimality': 'exact-enumeration',
        'dependency_closed': True,
        'upper_bound_equals_optimum': True,
        'boundary': 'exact solver for small dependency graphs; no rule matching, learned policy, future-hit oracle, or approximation claim',
    }
