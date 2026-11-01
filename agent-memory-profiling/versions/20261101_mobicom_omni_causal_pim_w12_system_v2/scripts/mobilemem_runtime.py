"""Label-free runtime contracts for temporal MobileMem execution."""
from __future__ import annotations

import numpy as np
from pim_backend.agent_memory_pim import plan_materialized_scan as plan_resident_scan
from trajectory_hypergraph import trajectory_unions as _trajectory_unions


def last_valid_positions(attention_mask):
    import torch

    if attention_mask.ndim != 2 or not bool(attention_mask.any(dim=1).all()):
        raise ValueError('every embedding row must have a visible token')
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values


def select_development(questions, *, limit, sample_count=0, shard_count=1, shard_index=0):
    if limit < 1 or sample_count < 0 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('invalid development selection')
    ordered = sorted((question for question in questions if question['split'] == 'development'),
                     key=lambda question: (question['effective_timestamp'], question['question_id']))[:limit]
    if sample_count:
        positions = np.linspace(0, len(ordered) - 1, min(sample_count, len(ordered)), dtype=int)
        ordered = [ordered[int(position)] for position in positions]
    start = len(ordered) * shard_index // shard_count
    end = len(ordered) * (shard_index + 1) // shard_count
    return ordered[start:end]


def hybrid_candidates(atoms, question, dense_indices, *, count=64):
    del question
    event_order, event_seen = [], set()
    for index in dense_indices:
        event_id = atoms[int(index)].get('event_id', int(index))
        if event_id not in event_seen:
            event_order.append(event_id)
            event_seen.add(event_id)
        if len(event_order) >= count:
            break
    selected = [index for index, atom in enumerate(atoms)
                if atom.get('event_id', index) in event_seen]
    rank = {event_id: position for position, event_id in enumerate(event_order)}
    selected.sort(key=lambda index: (rank[atoms[index].get('event_id', index)], index))
    return selected, {'dense_count': len(dense_indices), 'candidate_source': 'frozen dense event index',
                      'dense_top_indices': [int(index) for index in dense_indices],
                      'dense_top_events': event_order,
                      'candidate_granularity': 'event with all visible atom windows',
                      'boundary': 'candidate generation only; no lexical rule or label-driven fusion'}


def dialogue_unions(graph, visible, primary_indices):
    return _trajectory_unions(graph, visible, primary_indices)[0]


def trajectory_bundles(graph, visible, primary_indices, question):
    return _trajectory_unions(graph, visible, primary_indices, question)
