"""Label-free candidate-pool export used by offline calibration only."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path


SCHEMA = 'omni-candidate-dump-v1'


def _identity(value, name):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or value in {'.', '..'}):
        raise ValueError(f'valid {name} identity required')
    return value


def object_session_ids(objects, object_ids):
    """Read every session identity attached to the referenced source objects."""
    if not isinstance(objects, Mapping):
        raise ValueError('source objects must be a mapping')
    if (not isinstance(object_ids, Sequence) or isinstance(object_ids, (str, bytes))
            or not object_ids):
        raise ValueError('candidate object_ids must be a nonempty list')
    sessions = set()
    for object_id in object_ids:
        if not isinstance(object_id, str) or object_id not in objects:
            raise ValueError(f'candidate references unknown source object: {object_id}')
        obj = objects[object_id]
        if not isinstance(obj, Mapping):
            raise ValueError('source object must be a mapping')
        if obj.get('kind') == 'text':
            session_id = obj.get('session_id')
            if not isinstance(session_id, str) or not session_id:
                raise ValueError('text source object session missing')
            sessions.add(session_id)
        elif obj.get('kind') == 'image':
            sources = obj.get('sources')
            if not isinstance(sources, list) or not sources:
                raise ValueError('image source object sources missing')
            for source in sources:
                session_id = source.get('session_id') if isinstance(source, Mapping) else None
                if not isinstance(session_id, str) or not session_id:
                    raise ValueError('image source session missing')
                sessions.add(session_id)
        else:
            raise ValueError(f'unknown source object kind: {obj.get("kind")}')
    return sorted(sessions)


def _modality(record):
    kind = record.get('kind')
    if kind == 'source_text':
        return 'text'
    if kind == 'image_locator':
        return 'image'
    raise ValueError(f'unsupported candidate record kind: {kind}')


def _finite_score(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def _integer(value, name, *, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def build_candidate_dump(*, person_id, question_id, split, demands, candidates,
                         witnesses, objects, candidate_width=48):
    """Build one question's fixed-width candidate and accepted-witness export."""
    person_id = _identity(person_id, 'person')
    question_id = _identity(question_id, 'question')
    split = _identity(split, 'split')
    if (isinstance(candidate_width, bool) or not isinstance(candidate_width, int)
            or candidate_width <= 0):
        raise ValueError('positive candidate width required')
    if not isinstance(demands, list) or not isinstance(candidates, list):
        raise ValueError('demands and candidates must be lists')
    demand_ids = []
    for demand in demands:
        if not isinstance(demand, Mapping):
            raise ValueError('demand must be a mapping')
        demand_id = _identity(demand.get('demand_id'), 'demand')
        if demand_id in demand_ids:
            raise ValueError('duplicate demand identity')
        demand_ids.append(demand_id)
    grouped = {demand_id: [] for demand_id in demand_ids}
    by_demand_record = {demand_id: {} for demand_id in demand_ids}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError('candidate must be a mapping')
        demand_id = candidate.get('demand_id')
        if demand_id not in grouped:
            raise ValueError('candidate belongs to an unknown demand')
        record_id = _identity(candidate.get('base_record_id'), 'record')
        if record_id in by_demand_record[demand_id]:
            raise ValueError('duplicate candidate record identity')
        by_demand_record[demand_id][record_id] = candidate
        grouped[demand_id].append(candidate)

    exported_demands = []
    global_record_ids = set()
    for demand_id in demand_ids:
        rows = grouped[demand_id]
        if len(rows) != candidate_width:
            raise ValueError(
                f'demand {demand_id} has {len(rows)} candidates, '
                f'expected {candidate_width}')
        retrieval_ranks = [
            _integer(item.get('retrieval_rank'), 'retrieval_rank')
            for item in rows]
        rows = [row for _, row in sorted(
            zip(retrieval_ranks, rows), key=lambda item: item[0])]
        retrieval_ranks = sorted(retrieval_ranks)
        if retrieval_ranks != list(range(candidate_width)):
            raise ValueError('retrieval ranks must be unique and contiguous from zero')
        query_ranks = [
            _integer(item.get('query_rank'), 'query_rank') for item in rows]
        if len(set(query_ranks)) != len(query_ranks):
            raise ValueError('query ranks must be unique nonnegative integers')
        exported = []
        for row in rows:
            object_ids = row.get('object_ids')
            if (not isinstance(object_ids, list) or not object_ids
                    or any(not isinstance(object_id, str) or not object_id
                           for object_id in object_ids)):
                raise ValueError('candidate object_ids must be a nonempty list')
            exported.append({
                'record_id': str(row['base_record_id']),
                'object_ids': list(object_ids),
                'session_ids': object_session_ids(objects, object_ids),
                'modality': _modality(row),
                'retrieval_cosine': _finite_score(
                    row.get('retrieval_cosine'), 'retrieval_cosine'),
                'rerank_score': _finite_score(row.get('score'), 'rerank_score'),
                'query_rank': _integer(row['query_rank'], 'query_rank'),
                'retrieval_rank': _integer(row['retrieval_rank'], 'retrieval_rank'),
            })
        exported_demands.append({
            'demand_id': demand_id,
            'candidate_count': len(exported),
            'candidates': exported,
        })
        global_record_ids.update(item['record_id'] for item in exported)

    if not isinstance(witnesses, list):
        raise ValueError('witnesses must be a list')
    exported_witnesses = []
    witness_ids = set()
    for witness in witnesses:
        if not isinstance(witness, Mapping):
            raise ValueError('accepted witness must be a mapping')
        witness_id = _identity(witness.get('witness_id'), 'witness')
        if witness_id in witness_ids:
            raise ValueError('duplicate accepted witness identity')
        witness_ids.add(witness_id)
        witness_demand = _identity(witness.get('demand_id'), 'witness demand')
        if witness_demand not in by_demand_record:
            raise ValueError('accepted witness belongs to an unknown demand')
        record_ids = witness.get('retrieval_record_ids')
        if not isinstance(record_ids, list) or not record_ids:
            raise ValueError('accepted witness record_ids must reference candidates')
        invalid_record_ids = [record_id for record_id in record_ids
                              if not isinstance(record_id, str)]
        missing_record_ids = [record_id for record_id in record_ids
                              if isinstance(record_id, str)
                              and record_id not in global_record_ids]
        if invalid_record_ids or missing_record_ids:
            raise ValueError(
                'accepted witness record_ids must reference candidates: '
                f'demand={witness_demand} invalid={invalid_record_ids} '
                f'missing={missing_record_ids} '
                f'candidate_records={len(global_record_ids)}')
        object_ids = witness.get('object_ids')
        if (not isinstance(object_ids, list) or not object_ids
                or any(not isinstance(object_id, str) or not object_id
                       for object_id in object_ids)):
            raise ValueError('accepted witness object_ids must be a nonempty list')
        exported_witnesses.append({
            'witness_id': witness_id,
            'demand_id': witness_demand,
            'record_ids': list(record_ids),
            'proposal_record_ids': list(witness.get('record_ids') or []),
            'object_ids': list(object_ids),
            'base_record_id': witness.get('base_record_id'),
            'score': _finite_score(witness.get('score'), 'witness score'),
        })
    return {
        'schema': SCHEMA,
        'person_id': person_id,
        'question_id': question_id,
        'split': split,
        'label_free': True,
        'evaluation_label_files_read': [],
        'candidate_width': candidate_width,
        'demand_count': len(exported_demands),
        'demands': exported_demands,
        'witnesses': exported_witnesses,
        'boundary': ('serving-side candidate pool and accepted source witnesses; '
                     'no answer, evidence label, question type, or difficulty'),
    }


def write_candidate_dump(root, payload):
    """Atomically write one fresh ``<root>/<user>/<qid>.json`` dump."""
    if not isinstance(payload, Mapping):
        raise ValueError('candidate dump payload must be a mapping')
    person_id = _identity(payload.get('person_id'), 'person')
    question_id = _identity(payload.get('question_id'), 'question')
    path = Path(root) / person_id / f'{question_id}.json'
    if path.exists():
        raise FileExistsError(f'candidate dump already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    return path
