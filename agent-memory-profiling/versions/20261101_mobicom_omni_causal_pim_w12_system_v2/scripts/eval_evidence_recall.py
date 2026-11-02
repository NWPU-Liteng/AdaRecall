"""Label-side evidence recall for MobileMem-Omni serving outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'omni-evidence-recall-v1'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f'JSONL row must be an object: {path}:{number}')
        rows.append(value)
    return rows


def load_labels(path):
    labels = {}
    for label in read_jsonl(path):
        question_id = label.get('question_id')
        person_id = label.get('person_id')
        if not isinstance(question_id, str) or person_id is None:
            raise ValueError(f'label identity missing in {path}')
        identity = (str(person_id), question_id)
        if identity in labels:
            raise ValueError(f'duplicate label: {identity}')
        labels[identity] = label
    if not labels:
        raise ValueError(f'labels file is empty: {path}')
    return labels


def load_graph(labels_path, graph_path=None):
    labels_path = Path(labels_path)
    graph_path = Path(graph_path) if graph_path else labels_path.parent / 'omni_graph.json'
    if not graph_path.is_file():
        raise FileNotFoundError(f'omni graph not found: {graph_path}')
    audit_path = labels_path.parent / 'audit.json'
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        expected = audit.get('artifacts_sha256', {})
        if expected.get('labels.jsonl') and digest(labels_path) != expected['labels.jsonl']:
            raise ValueError('labels.jsonl does not match prepared audit')
        if expected.get('omni_graph.json') and digest(graph_path) != expected['omni_graph.json']:
            raise ValueError('omni_graph.json does not match prepared audit')
    payload = json.loads(graph_path.read_text())
    if not isinstance(payload, list):
        raise ValueError('omni graph must be a user array')
    graphs = {}
    for graph in payload:
        if not isinstance(graph, dict) or graph.get('person_id') is None:
            raise ValueError('omni graph user entry is malformed')
        person_id = str(graph['person_id'])
        if person_id in graphs:
            raise ValueError(f'duplicate omni graph user: {person_id}')
        if graph.get('label_free') is not True:
            raise ValueError(f'omni graph is not marked label_free: {person_id}')
        sessions = graph.get('sessions')
        if not isinstance(sessions, list):
            raise ValueError(f'omni graph sessions missing: {person_id}')
        session_ids = {str(session.get('session_id')) for session in sessions}
        if None in session_ids or 'None' in session_ids:
            raise ValueError(f'omni graph session identity missing: {person_id}')
        graphs[person_id] = session_ids
    return graphs, graph_path


def load_run(run_dir):
    run_dir = Path(run_dir)
    paths = [path for path in sorted(run_dir.glob('user_*_shard_*.json'))
             if not path.name.endswith('.partial.json')]
    if not paths:
        raise FileNotFoundError(f'no complete user shard outputs in {run_dir}')
    rows = []
    identities = set()
    index_hashes = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get('complete') is not True:
            raise ValueError(f'incomplete user shard output: {path}')
        person_id = payload.get('person_id')
        index_sha256 = payload.get('index_sha256')
        values = payload.get('rows')
        if person_id is None or not isinstance(index_sha256, str) or not index_sha256:
            raise ValueError(f'run shard identity missing: {path}')
        if not isinstance(values, list):
            raise ValueError(f'rows must be a list: {path}')
        person_id = str(person_id)
        previous = index_hashes.setdefault(person_id, index_sha256)
        if previous != index_sha256:
            raise ValueError(f'inconsistent source index for user {person_id}')
        for row in values:
            if not isinstance(row, dict):
                raise ValueError(f'row must be an object: {path}')
            if str(row.get('person_id')) != person_id:
                raise ValueError(f'row person mismatch: {path}')
            question_id = row.get('question_id')
            if not isinstance(question_id, str):
                raise ValueError(f'row question identity missing: {path}')
            identity = (person_id, question_id)
            if identity in identities:
                raise ValueError(f'duplicate question output: {identity}')
            identities.add(identity)
            rows.append(row)
    if not rows:
        raise ValueError(f'run contains no question rows: {run_dir}')
    return paths, rows, index_hashes


def _source_index_path(root, user_id):
    return Path(root) / f'proof_index_user_{user_id}' / 'source.json'


def resolve_source_indexes(labels_path, index_root, user_ids):
    labels_path = Path(labels_path)
    user_ids = sorted(str(value) for value in user_ids)

    def complete(root):
        return all(_source_index_path(root, user).is_file()
                   and _source_index_path(root, user).with_name('index.json').is_file()
                   for user in user_ids)

    if index_root:
        root = Path(index_root)
        if root.is_file() or root.name.startswith('proof_index_user_'):
            root = root if root.is_dir() else root.parent
            if root.name.startswith('proof_index_user_') and len(user_ids) == 1:
                root = root.parent
        if not complete(root):
            raise FileNotFoundError(
                f'incomplete source index under explicit root: {root}')
        selected = root
    else:
        candidates = []
        for candidate in (
                labels_path.parent.parent / 'indexes_final',
                labels_path.parent.parent / 'indexes',
                labels_path.parent / 'indexes_final',
                labels_path.parent / 'indexes'):
            if candidate not in candidates and complete(candidate):
                candidates.append(candidate)
        if not candidates:
            raise FileNotFoundError(
                'no complete source index found; pass --index-root')
        if len(candidates) != 1:
            raise ValueError(
                'ambiguous source index roots: ' + ', '.join(map(str, candidates)))
        selected = candidates[0]
    return selected, {user: _source_index_path(selected, user) for user in user_ids}


def load_sources(source_paths, run_index_hashes):
    sources = {}
    for user_id, path in source_paths.items():
        if user_id not in run_index_hashes:
            raise ValueError(f'run has no shard for user {user_id}')
        index_path = path.with_name('index.json')
        if digest(index_path) != run_index_hashes[user_id]:
            raise ValueError(f'source index does not match run shard: user {user_id}')
        payload = json.loads(path.read_text())
        if (payload.get('schema') != 'source-object-container-v2'
                or str(payload.get('person_id')) != user_id):
            raise ValueError(f'source index identity mismatch: {path}')
        objects = payload.get('objects')
        if not isinstance(objects, dict):
            raise ValueError(f'source objects missing: {path}')
        sources[user_id] = {
            'path': path,
            'sha256': digest(path),
            'index_sha256': run_index_hashes[user_id],
            'objects': objects,
        }
    return sources


def _image_ref_keys(ref):
    if not isinstance(ref, dict):
        return []
    values = []
    for name in ('raw', 'normalized', 'resolved'):
        value = ref.get(name)
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    return values


def build_image_ref_map(objects):
    mapping = {}
    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            raise ValueError(f'source object must be an object: {object_id}')
        if obj.get('kind') != 'image':
            continue
        sha256 = obj.get('source_sha256')
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f'image object sha256 missing: {object_id}')
        refs = [obj.get('image_ref')]
        sources = obj.get('sources')
        if not isinstance(sources, list) or not sources:
            raise ValueError(f'image object sources missing: {object_id}')
        refs.extend(source.get('image_ref') for source in sources
                    if isinstance(source, dict))
        for ref in refs:
            for key in _image_ref_keys(ref):
                previous = mapping.setdefault(key, sha256)
                if previous != sha256:
                    raise ValueError(f'ambiguous image reference: {key}')
    return mapping


def _string_set(values, path):
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError(f'{path} must be a list of strings')
    return set(values)


def accepted_evidence(row, source, graph_sessions):
    plan = row.get('proof_plan')
    if not isinstance(plan, dict):
        raise ValueError(f"proof_plan missing for {row.get('question_id')}")
    object_ids = plan.get('accepted_object_ids')
    if not isinstance(object_ids, list) or any(not isinstance(value, str)
                                               for value in object_ids):
        raise ValueError(f"accepted_object_ids missing for {row.get('question_id')}")
    objects = source['objects']
    sessions = set()
    images = set()
    for object_id in object_ids:
        obj = objects.get(object_id)
        if not isinstance(obj, dict):
            raise ValueError(f'final evidence object is absent from source index: {object_id}')
        if obj.get('kind') == 'text':
            session_id = obj.get('session_id')
            if not isinstance(session_id, str):
                raise ValueError(f'text object session missing: {object_id}')
            if session_id not in graph_sessions:
                raise ValueError(f'text object session absent from graph: {session_id}')
            sessions.add(session_id)
        elif obj.get('kind') == 'image':
            sha256 = obj.get('source_sha256')
            if not isinstance(sha256, str):
                raise ValueError(f'image object sha256 missing: {object_id}')
            images.add(sha256)
            values = obj.get('sources')
            if not isinstance(values, list) or not values:
                raise ValueError(f'image object sources missing: {object_id}')
            for value in values:
                session_id = value.get('session_id') if isinstance(value, dict) else None
                if not isinstance(session_id, str) or session_id not in graph_sessions:
                    raise ValueError(f'image source session absent from graph: {object_id}')
                sessions.add(session_id)
        else:
            raise ValueError(f'unknown final evidence object kind: {object_id}')
    return sessions, images


def evaluate_question(label, row, source, graph_sessions, image_ref_map):
    labeled_sessions = _string_set(
        label.get('source_session_ids'), 'source_session_ids')
    image_refs = label.get('evidence_image_refs')
    if not isinstance(image_refs, list) or any(not isinstance(value, dict)
                                               for value in image_refs):
        raise ValueError('evidence_image_refs must be a list of objects')

    accepted_sessions, accepted_images = accepted_evidence(
        row, source, graph_sessions)
    unmapped_sessions = sorted(value for value in labeled_sessions
                               if value not in graph_sessions)
    labeled_images = set()
    unmapped_image_refs = []
    for ref in image_refs:
        matches = {image_ref_map[key] for key in _image_ref_keys(ref)
                   if key in image_ref_map}
        if len(matches) > 1:
            raise ValueError(f'label image reference is ambiguous: {ref}')
        if matches:
            labeled_images.update(matches)
        else:
            unmapped_image_refs.append(ref)

    session_hits = len(labeled_sessions & accepted_sessions)
    image_hits = len(labeled_images & accepted_images)
    mapped_sessions = len(labeled_sessions) - len(unmapped_sessions)
    mapped_images = len(labeled_images)
    session_recall = session_hits / mapped_sessions if mapped_sessions else None
    image_recall = (image_hits / len(labeled_images)
                    if labeled_images else None)
    question_type = label.get('question_type')
    if not isinstance(question_type, str) or not question_type:
        raise ValueError(f"question_type missing for {label.get('question_id')}")
    proof_status = row['proof_plan'].get('proof_status') or 'missing'
    failures = len(unmapped_sessions) + len(unmapped_image_refs)
    return {
        'person_id': str(label['person_id']),
        'question_id': label['question_id'],
        'proof_status': str(proof_status),
        'question_type': question_type,
        'labeled_session_ids': sorted(labeled_sessions),
        'accepted_session_ids': sorted(accepted_sessions),
        'labeled_image_sha256': sorted(labeled_images),
        'accepted_image_sha256': sorted(accepted_images),
        'session_labeled_units': len(labeled_sessions),
        'session_mapped_units': mapped_sessions,
        'image_labeled_units': mapped_images + len(unmapped_image_refs),
        'image_mapped_units': mapped_images,
        'session_hits': session_hits,
        'session_recall': session_recall,
        'image_hits': image_hits,
        'image_recall': image_recall,
        'mapping_failures': {
            'session_ids': unmapped_sessions,
            'image_refs': unmapped_image_refs,
            'total': failures,
        },
    }


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _recall_summary(rows, prefix):
    values = [row[f'{prefix}_recall'] for row in rows
              if row[f'{prefix}_recall'] is not None]
    labeled = sum(row[f'{prefix}_labeled_units'] for row in rows)
    mapped = sum(row[f'{prefix}_mapped_units'] for row in rows)
    hits = sum(row[f'{prefix}_hits'] for row in rows)
    return {
        'questions_with_labels': len(values),
        'labeled_units': labeled,
        'mapped_units': mapped,
        'hits': hits,
        'mean_per_question': _mean(values),
        'pooled_recall': hits / mapped if mapped else None,
    }


def aggregate(rows):
    failures = {
        name: sum(len(row['mapping_failures'][name]) for row in rows)
        for name in ('session_ids', 'image_refs')
    }
    failures['total'] = failures['session_ids'] + failures['image_refs']
    return {
        'questions': len(rows),
        'evidence_recall_session': _recall_summary(rows, 'session'),
        'evidence_recall_image': _recall_summary(rows, 'image'),
        'mapping_failures': failures,
    }


def _grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: aggregate(groups[name]) for name in sorted(groups)}


def build_report(run_dir, labels_path, index_root=None, graph_path=None):
    labels = load_labels(labels_path)
    graphs, graph_path = load_graph(labels_path, graph_path=graph_path)
    run_paths, rows, run_index_hashes = load_run(run_dir)
    source_root, source_paths = resolve_source_indexes(
        labels_path, index_root, run_index_hashes)
    sources = load_sources(source_paths, run_index_hashes)
    image_ref_maps = {user: build_image_ref_map(source['objects'])
                      for user, source in sources.items()}

    results = []
    for row in rows:
        person_id = str(row['person_id'])
        identity = (person_id, row['question_id'])
        label = labels.get(identity)
        if label is None:
            raise ValueError(f'run question has no label: {identity}')
        if person_id not in graphs:
            raise ValueError(f'run user has no omni graph: {person_id}')
        results.append(evaluate_question(
            label, row, sources[person_id], graphs[person_id],
            image_ref_maps[person_id]))

    overall = aggregate(results)
    by_proof_status = _grouped(results, 'proof_status')
    by_question_type = _grouped(results, 'question_type')
    cross = defaultdict(lambda: defaultdict(list))
    for result in results:
        cross[result['proof_status']][result['question_type']].append(result)
    by_both = {
        proof_status: {
            question_type: aggregate(values)
            for question_type, values in sorted(question_types.items())
        }
        for proof_status, question_types in sorted(cross.items())
    }
    source_provenance = {
        user: {
            'source_json': str(source['path']),
            'source_sha256': source['sha256'],
            'index_sha256': source['index_sha256'],
        }
        for user, source in sorted(sources.items())
    }
    return {
        'schema': SCHEMA,
        'run_dir': str(Path(run_dir).resolve()),
        'labels': str(Path(labels_path).resolve()),
        'graph': str(graph_path.resolve()),
        'source_root': str(Path(source_root).resolve()),
        'source_indexes': source_provenance,
        'source_files': [path.name for path in run_paths],
        'evidence_levels': {
            'session': 'source_session_ids mapped through accepted source objects',
            'image': 'evidence_image_refs mapped to source sha256',
            'atom': 'not_evaluated: labels do not provide atom_id',
            'event': 'not_evaluated: raw event ids do not identify final objects',
        },
        'questions': results,
        'evidence_recall_session': overall['evidence_recall_session'],
        'evidence_recall_image': overall['evidence_recall_image'],
        'mapping_failures': overall['mapping_failures'],
        'by_proof_status': by_proof_status,
        'by_question_type': by_question_type,
        'by_proof_status_and_question_type': by_both,
        'candidate_recall_session': {
            'status': 'skipped_no_candidate_dump',
            'reason': 'candidate dump is introduced by T3.1 and is absent from this run',
            'mean_per_question': None,
            'pooled_recall': None,
        },
        'boundary': 'label-side evaluation only; session/image recall does not establish semantic sufficiency or answer correctness',
    }


def write_report(run_dir, labels_path, *, index_root=None, graph_path=None,
                 output=None):
    report = build_report(run_dir, labels_path, index_root=index_root,
                          graph_path=graph_path)
    output = Path(output) if output else Path(run_dir) / 'evidence_recall.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                    sort_keys=True) + '\n')
    temporary.replace(output)
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--index-root', type=Path)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output, report = write_report(
        args.run_dir, args.labels, index_root=args.index_root,
        graph_path=args.graph, output=args.output)
    print(output)
    print(json.dumps({
        'questions': len(report['questions']),
        'evidence_recall_session': report['evidence_recall_session'],
        'evidence_recall_image': report['evidence_recall_image'],
        'mapping_failures': report['mapping_failures'],
        'candidate_recall_session': report['candidate_recall_session']['status'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
