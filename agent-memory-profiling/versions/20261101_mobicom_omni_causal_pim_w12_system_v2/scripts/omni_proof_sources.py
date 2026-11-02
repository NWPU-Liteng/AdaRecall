"""Separate observable containers, source-located records and physical objects."""
from __future__ import annotations

import json

from omni_prepare import parse_time
from omni_proof_runtime import checksum
from omni_provenance_optimizer import build_provenance_hyperedges


def timed_atom(atom):
    interval = f"{atom.get('visibility_start', 'unknown')}..{atom.get('visibility_end', 'unknown')}"
    return (f"[{atom['session_id']}][{atom['atom_id']}][{atom['role']}]"
            f"[session_interval={interval}; not_turn_timestamp] {atom['content']}")


def _session_visibility_end(session):
    values = []
    for field in ('visibility_end', 'end_time'):
        value = session.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f'session {field} must be a non-empty string')
        values.append((field, parse_time(value)))
    if not values:
        raise ValueError('session visibility_end/end_time missing')
    if any(value != values[0][1] for _, value in values[1:]):
        raise ValueError('session visibility_end and end_time disagree')
    return values[0][1]


def visible_sessions(graph, question):
    """Return sessions whose complete visibility interval ends by the question time."""
    if not isinstance(graph, dict) or not isinstance(graph.get('sessions'), list):
        raise ValueError('omni graph sessions missing')
    if not isinstance(question, dict):
        raise ValueError('question must be an object')
    timestamp = question.get('effective_timestamp')
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError('question effective_timestamp missing')
    cutoff = parse_time(timestamp)
    visible = set()
    for session in graph['sessions']:
        if not isinstance(session, dict):
            raise ValueError('omni graph session must be an object')
        session_id = session.get('session_id')
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('omni graph session identity missing')
        if session_id in visible:
            raise ValueError(f'duplicate omni graph session: {session_id}')
        if _session_visibility_end(session) <= cutoff:
            visible.add(session_id)
    return frozenset(visible)


def object_is_visible(obj, visible_session_ids):
    """Apply the text/image object visibility rule under a session set."""
    if not isinstance(obj, dict):
        raise ValueError('source object must be an object')
    visible_session_ids = frozenset(map(str, visible_session_ids))
    kind = obj.get('kind')
    if kind == 'text':
        session_id = obj.get('session_id')
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('text object session missing')
        return session_id in visible_session_ids
    if kind == 'image':
        sources = obj.get('sources')
        if not isinstance(sources, list) or not sources:
            raise ValueError('image object sources missing')
        session_ids = []
        for source in sources:
            session_id = source.get('session_id') if isinstance(source, dict) else None
            if not isinstance(session_id, str) or not session_id:
                raise ValueError('image source session missing')
            session_ids.append(session_id)
        return any(session_id in visible_session_ids for session_id in session_ids)
    raise ValueError(f'unknown source object kind: {kind}')


def build_sources(graph, image_hashes):
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    sessions = {}
    for atom in graph['atoms']:
        sessions.setdefault(atom['session_id'], []).append(atom)
    containers = build_provenance_hyperedges(list(sessions), sessions)
    objects, records, missing = {}, [], []
    for container in containers:
        text_ids = []
        for identity in container['atom_ids']:
            atom = atoms[identity]
            if atom.get('content_type') == 'text' and atom.get('content'):
                object_id = 'text:' + identity
                objects[object_id] = {'object_id': object_id, 'kind': 'text',
                    'atom_id': identity, 'session_id': atom['session_id'], 'role': atom['role'],
                    'ordinal': atom['ordinal'], 'source_sha256': checksum(atom['content']),
                    'text': timed_atom(atom), 'raw_text': atom['content'],
                    'source_span': [0, len(atom['content'])],
                    'time': [atom.get('visibility_start'), atom.get('visibility_end')]}
                text_ids.append(object_id)
        ordered = sorted((atoms[identity] for identity in container['atom_ids']),
                         key=lambda atom: atom['ordinal'])

        def nearby_text(identity):
            position = next(index for index, atom in enumerate(ordered) if atom['atom_id'] == identity)
            previous = next((atom for atom in reversed(ordered[:position])
                             if 'text:' + atom['atom_id'] in objects), None)
            following = next((atom for atom in ordered[position + 1:]
                              if 'text:' + atom['atom_id'] in objects), None)
            return ['text:' + atom['atom_id'] for atom in (previous, following) if atom]

        for object_id in text_ids:
            obj = objects[object_id]
            records.append({'record_id': 'atom:' + obj['atom_id'], 'kind': 'source_text',
                'container_id': container['edge_id'], 'primary_object_id': object_id,
                'object_ids': [object_id], 'retrieval_context_object_ids': nearby_text(obj['atom_id']),
                'source_atom_ids': [obj['atom_id']], 'source_span': obj['source_span'],
                'session_id': container['session_id'], 'locator': obj['text'],
                'status': 'original_role_attributed_text_not_verified_world_fact'})
        for identity in container['atom_ids']:
            atom = atoms[identity]
            for ref in atom.get('image_refs', []):
                member = ref.get('resolved', ref['normalized'])
                if ref.get('status') not in {'exact', 'unique_user_basename'} or member not in image_hashes:
                    missing.append({'atom_id': identity, 'image_ref': ref})
                    continue
                object_id = 'image:' + image_hashes[member]
                objects.setdefault(object_id, {'object_id': object_id, 'kind': 'image',
                    'source_sha256': image_hashes[member], 'image_ref': ref, 'sources': []})
                source = {'atom_id': identity, 'session_id': atom['session_id'],
                    'role': atom['role'], 'ordinal': atom['ordinal'],
                    'time': [atom.get('visibility_start'), atom.get('visibility_end')],
                    'image_ref': ref}
                if source not in objects[object_id]['sources']:
                    objects[object_id]['sources'].append(source)
                text_object_ids = nearby_text(identity)
                records.append({'record_id': 'visual:' + checksum(container['edge_id'] + identity + member),
                    'kind': 'image_locator', 'container_id': container['edge_id'],
                    'primary_object_id': object_id,
                    'object_ids': [object_id], 'image_object_id': object_id,
                    'retrieval_context_object_ids': text_object_ids,
                    'source_atom_ids': [objects[value]['atom_id'] for value in text_object_ids] + [identity],
                    'session_id': atom['session_id'], 'image_source': source,
                    'context': '\n'.join(objects[value]['text'] for value in text_object_ids),
                    'status': 'model_locator_only_original_image_required'})
    return {'schema': 'source-object-container-v2', 'person_id': graph['person_id'],
            'containers': containers, 'objects': objects, 'records': records,
            'unresolved_sources': missing,
            'boundary': 'one primary atom per text record; neighboring turns are optional source context, not forced materialization; session time is not per-turn event time'}


def materialize(objects, object_ids, source_hints=None, visible_session_ids=None):
    source_hints = source_hints or {}
    identities = sorted(set(object_ids))
    selected = [objects[identity] for identity in identities]
    if visible_session_ids is not None:
        visible_session_ids = frozenset(map(str, visible_session_ids))
        forbidden = [identity for identity, value in zip(identities, selected)
                     if not object_is_visible(value, visible_session_ids)]
        if forbidden:
            raise ValueError('attempted to materialize invisible object(s): '
                             + ','.join(forbidden))
    texts = sorted((value for value in selected if value['kind'] == 'text'),
                   key=lambda value: (value['ordinal'], value['object_id']))
    images = sorted((value for value in selected if value['kind'] == 'image'),
                    key=lambda value: (value['sources'][0]['ordinal'], value['object_id']))
    refs = []
    for image in images:
        sources = image['sources']
        if visible_session_ids is not None:
            sources = [source for source in sources
                       if str(source['session_id']) in visible_session_ids]
        source = source_hints.get(image['object_id'])
        if (source is None
                or (visible_session_ids is not None
                    and str(source.get('session_id')) not in visible_session_ids)):
            source = sources[0]
        refs.append({**source['image_ref'], 'source_atom_id': source['atom_id'],
                     'source_session_id': source['session_id'], 'object_id': image['object_id'],
                     'image_sha256': image['source_sha256'], 'all_observable_sources': sources})
    return {'text': '\n'.join(value['text'] for value in texts), 'images': refs,
            'accepted_atom_ids': [value['atom_id'] for value in texts],
            'materialized_object_ids': sorted(set(object_ids))}


def image_source_header(ref):
    sources = [{key: source[key] for key in ('atom_id', 'session_id', 'role', 'time')}
               for source in ref['all_observable_sources']]
    return ('原始图片，可见来源=' + json.dumps(sources, ensure_ascii=False, separators=(',', ':'))
            + '；time是session可见范围而非行为时刻，只把图中可见内容作为证据。')
