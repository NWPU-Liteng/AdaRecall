"""Compact candidate frontier for source-grounding backend planning."""
from __future__ import annotations

import hashlib
import json
import math


FRONTIER_SCHEMA = 'evidence-candidate-frontier-v1'


def _finite_score(value, field):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{field} must be finite')
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _normalize_bindings(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [[str(key), str(item)] for key, item in sorted(value.items())]
    if not isinstance(value, list):
        raise ValueError('candidate bindings must be a mapping or list')
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append({key: item[key] for key in sorted(item)})
        else:
            normalized.append(item)
    return sorted(normalized, key=_canonical)


def _normalize_relations(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError('candidate source relations must be a list')
    normalized = []
    for item in value:
        if isinstance(item, dict):
            normalized.append({key: item[key] for key in sorted(item)})
        else:
            normalized.append(item)
    return sorted(normalized, key=_canonical)


def _relation_identity(value):
    if isinstance(value, dict):
        value = value.get('relation_id')
    return value if isinstance(value, (str, int, float, bool, type(None))) else _canonical(value)


def _candidate_record(candidate, index):
    if not isinstance(candidate, dict):
        raise ValueError('candidate records must be mappings')
    demand_id = candidate.get('demand_id')
    base_record_id = candidate.get('base_record_id', candidate.get('record_id'))
    object_ids = candidate.get('object_ids')
    context_object_ids = candidate.get('retrieval_context_object_ids', [])
    retrieval_text = candidate.get('retrieval_text', '')
    if not isinstance(demand_id, str) or not demand_id:
        raise ValueError('candidate demand identity required')
    if not isinstance(base_record_id, str) or not base_record_id:
        raise ValueError('candidate source identity required')
    if not isinstance(object_ids, list) or not object_ids:
        raise ValueError('candidate object identities required')
    if len(set(object_ids)) != len(object_ids):
        raise ValueError('candidate object identities must be unique')
    if not isinstance(context_object_ids, list):
        raise ValueError('candidate retrieval context identities must be a list')
    if len(set(context_object_ids)) != len(context_object_ids):
        raise ValueError('candidate retrieval context identities must be unique')
    if not isinstance(retrieval_text, str):
        raise ValueError('candidate retrieval text must be a string')
    score = _finite_score(candidate.get('score'), 'candidate score')
    bindings = _normalize_bindings(candidate.get('bindings'))
    source_relations = _normalize_relations(candidate.get('source_relations'))
    semantic_payload = {
        'demand_id': demand_id,
        'base_record_id': base_record_id,
        'object_ids': list(object_ids),
        'retrieval_context_object_ids': list(context_object_ids),
        'bindings': bindings,
        'source_relations': source_relations,
    }
    semantic_key = hashlib.sha256(_canonical(semantic_payload).encode('utf-8')).hexdigest()
    witness_id = candidate.get('witness_id')
    if not isinstance(witness_id, str) or not witness_id:
        witness_id = f'{base_record_id}#demand={demand_id}#candidate={index}'
    descriptor = {
        'demand_id': demand_id,
        'base_record_id': base_record_id,
        'witness_id': witness_id,
        'object_ids': list(object_ids),
        'retrieval_context_object_ids': list(context_object_ids),
        'query_rank': int(candidate.get('query_rank', 0)),
        'score': score,
        'retrieval_cosine': (_finite_score(candidate['retrieval_cosine'], 'retrieval cosine')
                             if 'retrieval_cosine' in candidate else None),
        'bindings': bindings,
        'source_relations': source_relations,
        'semantic_key': semantic_key,
        'candidate_index': index,
        'descriptor_bytes': 96 + len(retrieval_text.encode('utf-8'))
                           + len(_canonical(semantic_payload).encode('utf-8')),
    }
    descriptor['_dedup_key'] = _canonical(semantic_payload)
    return descriptor


def compile_candidate_frontier(candidates, *, question_id=None):
    """Compile a lossless descriptor frontier for adaptive grounding.

    Only exact semantic duplicates are removed.  Candidates with different
    source identity, object coverage, bindings or source relations remain
    separate, because collapsing any of those dimensions can remove a valid
    AND/OR witness or alter provenance.
    """
    if not isinstance(candidates, list):
        raise ValueError('candidate list required')
    unique = {}
    for index, candidate in enumerate(candidates):
        record = _candidate_record(candidate, index)
        previous = unique.get(record['_dedup_key'])
        if previous is None or (record['score'], -record['candidate_index']) > (
                previous['score'], -previous['candidate_index']):
            unique[record['_dedup_key']] = record
    records = sorted(unique.values(), key=lambda record: (
        record['demand_id'], -record['score'], record['base_record_id']))
    for record in records:
        record.pop('_dedup_key', None)
    demand_counts = {}
    for record in records:
        demand_counts[record['demand_id']] = demand_counts.get(record['demand_id'], 0) + 1
    packed_bytes = sum(record['descriptor_bytes'] for record in records)
    source_ids = {record['base_record_id'] for record in records}
    relation_ids = set()
    binding_keys = set()
    for record in records:
        for relation in record['source_relations']:
            relation_ids.add(_relation_identity(relation))
        for binding in record['bindings']:
            if isinstance(binding, dict):
                binding_keys.update(binding)
            elif isinstance(binding, list) and binding:
                binding_keys.add(binding[0])
    return {
        'schema': FRONTIER_SCHEMA,
        'question_id': question_id,
        'records': records,
        'input_candidate_count': len(candidates),
        'state_count': len(records),
        'demand_count': len(demand_counts),
        'candidate_count_by_demand': demand_counts,
        'source_provenance_count': len(source_ids),
        'source_relation_count': len(relation_ids),
        'binding_key_count': len(binding_keys),
        'packed_bytes': packed_bytes,
        'payload_bytes': packed_bytes,
        'metadata_bytes': 0,
        'deduplicated_candidate_count': len(candidates) - len(records),
        'semantic_pruning_applied': False,
        'boundary': 'lossless descriptor frontier; no factual sufficiency or PIM execution claim',
    }


def build_frontier_levels(frontier, *, seed_per_demand=0, growth_factor=2):
    """Build monotonically expanding candidate levels with coverage preserved.

    The first level is an optimization opportunity, not a semantic deletion:
    every later level is retained and the final level is the complete frontier.
    A caller must fall back to the next level when the grounding contract is
    incomplete.  A zero seed selects the complete frontier directly.
    """
    if frontier.get('schema') != FRONTIER_SCHEMA:
        raise ValueError('candidate frontier schema required')
    if type(seed_per_demand) is not int or seed_per_demand < 0:
        raise ValueError('seed_per_demand must be a nonnegative integer')
    if type(growth_factor) is not int or growth_factor < 2:
        raise ValueError('growth_factor must be an integer of at least two')
    records = list(frontier.get('records', []))
    by_demand = {}
    for record in records:
        by_demand.setdefault(record['demand_id'], []).append(record)
    for values in by_demand.values():
        values.sort(key=lambda item: (-item['score'], item['query_rank'], item['base_record_id']))
    max_per_demand = max((len(values) for values in by_demand.values()), default=0)
    if not max_per_demand:
        return []
    quotas = [max_per_demand]
    if seed_per_demand:
        quotas = [min(seed_per_demand, max_per_demand)]
        while quotas[-1] < max_per_demand:
            quotas.append(min(max_per_demand, quotas[-1] * growth_factor))
    levels = []
    def contract_key(record):
        return (_canonical(record.get('bindings', [])),
                _canonical(record.get('source_relations', [])))

    def select_domain(values, quota):
        representatives = {}
        for record in values:
            representatives.setdefault(contract_key(record), record)
        mandatory = sorted(representatives.values(),
                           key=lambda item: (-item['score'], item['query_rank'], item['base_record_id']))
        target = max(quota, len(mandatory))
        selected = mandatory[:target]
        selected_ids = {record['witness_id'] for record in selected}
        selected.extend(record for record in values
                        if record['witness_id'] not in selected_ids
                        and len(selected) < target)
        return selected

    for level_index, quota in enumerate(quotas):
        selected = [record for demand_id in sorted(by_demand)
                    for record in select_domain(by_demand[demand_id], quota)]
        selected_ids = {record['witness_id'] for record in selected}
        demand_coverage = {demand_id: sum(record['demand_id'] == demand_id
                                          for record in selected)
                           for demand_id in sorted(by_demand)}
        source_ids = {record['base_record_id'] for record in selected}
        relation_ids = set()
        binding_keys = set()
        for record in selected:
            for relation in record.get('source_relations', []):
                relation_ids.add(_relation_identity(relation))
            for binding in record.get('bindings', []):
                if isinstance(binding, dict):
                    binding_keys.update(binding)
                elif isinstance(binding, list) and binding:
                    binding_keys.add(binding[0])
        full_source_ids = {record['base_record_id'] for record in records}
        full_relation_ids = set()
        full_binding_keys = set()
        for record in records:
            for relation in record.get('source_relations', []):
                full_relation_ids.add(_relation_identity(relation))
            for binding in record.get('bindings', []):
                if isinstance(binding, dict):
                    full_binding_keys.update(binding)
                elif isinstance(binding, list) and binding:
                    full_binding_keys.add(binding[0])
        levels.append({
            'level': level_index,
            'quota_per_demand': quota,
            'selected_witness_ids': sorted(selected_ids),
            'selected_candidate_indices': sorted(record['candidate_index'] for record in selected),
            'candidate_count': len(selected),
            'demand_coverage': demand_coverage,
            'covered_demand_count': sum(bool(count) for count in demand_coverage.values()),
            'source_provenance_count': len(source_ids),
            'lost_source_provenance_count': len(full_source_ids - source_ids),
            'source_relation_count': len(relation_ids),
            'lost_source_relation_count': len(full_relation_ids - relation_ids),
            'binding_key_count': len(binding_keys),
            'lost_binding_key_count': len(full_binding_keys - binding_keys),
            'packed_bytes': sum(record['descriptor_bytes'] for record in selected),
            'is_complete': len(selected_ids) == len(records),
            'selection_policy': 'contract-class representatives then score order',
        })
    return levels


def candidates_for_frontier_level(candidates, level):
    """Return original candidate mappings selected by a frontier level."""
    if not isinstance(candidates, list) or not isinstance(level, dict):
        raise ValueError('candidate list and frontier level required')
    indices = level.get('selected_candidate_indices')
    if not isinstance(indices, list):
        raise ValueError('frontier level candidate indices required')
    return [candidates[index] for index in indices if 0 <= index < len(candidates)]
