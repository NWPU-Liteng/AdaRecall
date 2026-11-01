"""Compile observable dialogue hyperedges without inferred factual causality."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json


def augment_graph(graph: dict) -> dict:
    result = dict(graph)
    result['hypergraph_contract'] = {
        'edge_kinds': sorted({edge['kind'] for edge in graph.get('hyperedges', [])}),
        'hard_edge_kinds': ['dialogue_dependency'],
        'context_space_semantics': 'recorded application/session identity, not physical space',
        'temporal_semantics': 'recorded order and visibility, not inferred world-state updates',
        'label_free': True,
    }
    return result


def trajectory_unions(graph, visible, primary_indices, question='', **unused):
    del question, unused
    local_by_event = defaultdict(set)
    for index, atom in enumerate(visible):
        local_by_event[atom['event_id']].add(index)
    prerequisites = graph['dependencies']
    bundles, metadata = [], []
    for primary_index in primary_indices:
        seed = visible[primary_index]['event_id']
        required, frontier = {seed}, [seed]
        while frontier:
            event = frontier.pop()
            chain = {event}
            while event in prerequisites:
                parent = prerequisites[event]
                if parent in chain:
                    raise ValueError('observable dependency cycle')
                if parent not in local_by_event:
                    raise ValueError('observable prerequisite is not temporally visible')
                chain.add(parent)
                required.add(parent)
                event = parent
        bundle = frozenset(index for event in required for index in local_by_event[event])
        identity = hashlib.sha256(json.dumps(
            sorted(visible[index]['identity'] for index in bundle)).encode()).hexdigest()
        bundles.append(bundle)
        metadata.append({
            'identity': identity, 'seed_event_id': seed, 'closure_role': 'directed_observable_prerequisite_closure',
            'bundle_event_ids': sorted(required), 'mandatory_events': sorted(required),
            'closure_atoms': len(bundle), 'hard_dependency_closed': True,
            'edge_kinds': ['dialogue_dependency'] if len(required) > 1 else [],
            'dependency_direction': 'child_to_recorded_prerequisite',
        })
    return bundles, {'bundles': metadata, 'contract': 'observable-closure-materialization-v1',
                     'dependency_direction': 'child_to_recorded_prerequisite',
                     'boundary': 'directed provenance prerequisites, not future responses or factual entailment'}
