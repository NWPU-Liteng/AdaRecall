"""Label-free descriptor index for precomputed single-source witness branches."""
from __future__ import annotations

import hashlib
import json
import math

from omni_multi_witness import CANDIDATE_ONLY, UNVERIFIED, source_family


SCHEMA = 'witness-descriptor-index-v1'


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _descriptor_id(payload):
    return 'wd:' + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _image_hash(object_id):
    if not isinstance(object_id, str) or not object_id.startswith('image:'):
        raise ValueError('image descriptor requires an image object identity')
    value = object_id.split(':', 1)[1]
    if not value:
        raise ValueError('image object identity requires a sha256')
    return value


def descriptor_from_record(record, image_costs):
    """Convert one exact proof-index record into a candidate-only branch descriptor."""
    if not isinstance(record, dict):
        raise ValueError('source record must be an object')
    record_id = record.get('record_id')
    if not isinstance(record_id, str) or not record_id:
        raise ValueError('source record requires a nonempty record identity')
    kind = record.get('kind')
    if kind not in {'source_text', 'image_locator'}:
        raise ValueError(f'unsupported source record kind: {kind}')
    family = source_family(record_id)
    object_ids = record.get('object_ids')
    if (not family or not isinstance(object_ids, list) or not object_ids
            or any(not isinstance(value, str) or not value for value in object_ids)):
        raise ValueError(f'record {record_id} requires source family and object identities')
    primary_object_id = record.get('primary_object_id')
    if primary_object_id not in object_ids:
        raise ValueError(f'record {record_id} primary object is not in object_ids')
    source_atom_ids = record.get('source_atom_ids')
    if (not isinstance(source_atom_ids, list) or not source_atom_ids
            or any(not isinstance(value, str) or not value for value in source_atom_ids)):
        raise ValueError(f'record {record_id} requires source atom identities')

    if kind == 'source_text':
        cost = record.get('retrieval_tokens')
        if type(cost) is not int or cost < 0:
            raise ValueError(f'record {record_id} requires nonnegative retrieval_tokens')
        anchor = {'record_id': record_id, 'object_id': primary_object_id,
                  'text_index': 0, 'source_span': record.get('source_span')}
    else:
        image_hash = _image_hash(primary_object_id)
        cost = image_costs.get(image_hash)
        if type(cost) is not int or cost < 0:
            raise ValueError(f'record {record_id} lacks an exact Reader image cost')
        region_index = record.get('source_region_index')
        if type(region_index) is not int or region_index < 0:
            suffix = ':region:'
            if suffix not in record_id:
                raise ValueError(f'record {record_id} lacks a region identity')
            try:
                region_index = int(record_id.rsplit(suffix, 1)[1])
            except ValueError as error:
                raise ValueError(f'record {record_id} has an invalid region identity') from error
        anchor = {'record_id': record_id, 'object_id': primary_object_id,
                  'region_index': region_index, 'image_sha256': image_hash}

    payload = {
        'record_id': record_id,
        'source_family': family,
        'kind': kind,
        'object_ids': list(dict.fromkeys(object_ids)),
        'source_atom_ids': list(dict.fromkeys(source_atom_ids)),
        'anchor': anchor,
        'materialization_tokens': cost,
    }
    return {
        'descriptor_id': _descriptor_id(payload),
        **payload,
        'verification_state': CANDIDATE_ONLY,
        'semantic_sufficiency': UNVERIFIED,
    }


def compile_descriptor_index(records, image_costs, *, person_id, source_index_sha256,
                             records_sha256):
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty proof-index records are required')
    descriptors, seen = [], set()
    for record in records:
        descriptor = descriptor_from_record(record, image_costs)
        record_id = descriptor['record_id']
        if record_id in seen:
            raise ValueError(f'duplicate source record identity: {record_id}')
        seen.add(record_id)
        descriptors.append(descriptor)
    families = {item['source_family'] for item in descriptors}
    return {
        'schema': SCHEMA,
        'person_id': str(person_id),
        'source_index_sha256': source_index_sha256,
        'records_sha256': records_sha256,
        'label_free': True,
        'semantic_claims': False,
        'record_count': len(descriptors),
        'source_family_count': len(families),
        'kind_counts': {
            kind: sum(item['kind'] == kind for item in descriptors)
            for kind in ('source_text', 'image_locator')
        },
        'descriptors': descriptors,
        'boundary': ('precomputed source identities and costs only; descriptors are candidate branches, '
                     'not source-located or semantically sufficient witnesses'),
    }


def validate_descriptor_index(index, records, *, person_id, source_index_sha256,
                              records_sha256):
    if (not isinstance(index, dict) or index.get('schema') != SCHEMA
            or str(index.get('person_id')) != str(person_id)
            or index.get('source_index_sha256') != source_index_sha256
            or index.get('records_sha256') != records_sha256
            or index.get('label_free') is not True
            or index.get('semantic_claims') is not False):
        raise ValueError('witness descriptor index does not match the proof index')
    descriptors = index.get('descriptors')
    if not isinstance(descriptors, list) or len(descriptors) != len(records):
        raise ValueError('witness descriptor count does not match proof-index records')
    expected = {record['record_id'] for record in records}
    by_record = {}
    for item in descriptors:
        if not isinstance(item, dict) or item.get('record_id') not in expected:
            raise ValueError('witness descriptor has an unknown source identity')
        if item['record_id'] in by_record:
            raise ValueError('witness descriptor identities must be unique')
        if item.get('verification_state') != CANDIDATE_ONLY:
            raise ValueError('descriptor index cannot contain verified witnesses')
        if item.get('semantic_sufficiency') != UNVERIFIED:
            raise ValueError('descriptor index cannot claim semantic sufficiency')
        by_record[item['record_id']] = item
    if set(by_record) != expected or index.get('record_count') != len(records):
        raise ValueError('witness descriptor record set is incomplete')
    return by_record


def rank_source_family_frontier(records, scores, descriptors_by_record, *, limit,
                                required_kind=None, eligible_record_ids=None):
    """Return the best record per source family under a supplied score vector."""
    if not isinstance(records, list) or not isinstance(scores, (list, tuple)):
        raise ValueError('records and scores must be sequences')
    if len(records) != len(scores):
        raise ValueError('record and score counts must match')
    if type(limit) is not int or limit < 1:
        raise ValueError('positive source-family frontier limit required')
    if required_kind not in {None, 'source_text', 'image_locator'}:
        raise ValueError('unsupported evidence scope kind')
    eligible_record_ids = set(eligible_record_ids) if eligible_record_ids is not None else None
    best = {}
    for record, raw_score in zip(records, scores):
        record_id = record.get('record_id')
        score = float(raw_score)
        if not math.isfinite(score):
            continue
        if required_kind is not None and record.get('kind') != required_kind:
            continue
        if eligible_record_ids is not None and record_id not in eligible_record_ids:
            continue
        descriptor = descriptors_by_record.get(record_id)
        if descriptor is None:
            raise ValueError(f'missing witness descriptor for source record: {record_id}')
        family = descriptor['source_family']
        previous = best.get(family)
        if previous is None or (score, record_id) > (previous['score'], previous['record_id']):
            best[family] = {
                **descriptor,
                'score': score,
                'record_id': record_id,
                'record_ids': [record_id],
                'retrieval_record_ids': [record_id],
            }
    ranked = sorted(best.values(), key=lambda item: (
        -item['score'], item['source_family'], item['record_id']))[:limit]
    return {
        'schema': 'witness-descriptor-frontier-v1',
        'branches': ranked,
        'source_family_count': len(ranked),
        'candidate_or_available': len(ranked) >= 2,
        'semantic_claims': False,
        'verification_state': CANDIDATE_ONLY,
    }


def audit_descriptor_frontiers(frontiers):
    """Aggregate candidate-only family frontiers without upgrading their evidence state."""
    per_demand = []
    for demand_id, frontier in frontiers.items():
        branches = frontier.get('branches', [])
        per_demand.append({
            'demand_id': demand_id,
            'branch_count': len(branches),
            'source_family_count': frontier.get('source_family_count', len(branches)),
            'pre_rerank_family_count': frontier.get(
                'pre_rerank_family_count', frontier.get('source_family_count', len(branches))),
            'candidate_or_available': bool(frontier.get('candidate_or_available')),
            'families': [item['source_family'] for item in branches],
            'records': [item['record_id'] for item in branches],
            'verification_state': CANDIDATE_ONLY,
        })
    return {
        'demand_count': len(per_demand),
        'demands_with_candidate_only_family_or': sum(
            item['candidate_or_available'] for item in per_demand),
        'demands_with_semantically_verified_or': 0,
        'per_demand': per_demand,
        'boundary': 'candidate-only source-family diversity; no semantic sufficiency claim',
    }
