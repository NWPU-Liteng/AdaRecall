"""Evaluation-only development-split fit for fixed-width candidate pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from eval_evidence_recall import (_image_ref_keys, build_image_ref_map, digest,
                                  load_labels, resolve_source_indexes)


SCHEMA = 'omni-conformal-calibrator-v1'
CANDIDATE_WIDTH = 48
ALPHAS = (0.05, 0.1, 0.2)
PROBABILITY_CLIP = (1e-4, 1.0 - 1e-4)
COVERAGE_NOTE = '覆盖保证以问题可交换为前提，且只针对 48 个候选池。'


def _finite_number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def _identity(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} identity required')
    return value


def _string_list(value, name):
    if (not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)):
        raise ValueError(f'{name} must be a list of nonempty strings')
    return list(value)


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be a lowercase sha256')
    return value


def load_candidate_dumps(root):
    """Load and validate ``<root>/<user>/<question>.json`` development dumps."""
    root = Path(root)
    paths = sorted(root.glob('*/*.json'))
    if not paths:
        raise FileNotFoundError(f'no candidate dumps found under {root}')
    questions = {}
    users = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or payload.get('schema') != 'omni-candidate-dump-v1':
            raise ValueError(f'candidate dump schema mismatch: {path}')
        split = payload.get('split')
        if split != 'development':
            raise ValueError(f'candidate dump split must be development: {path}: {split}')
        if payload.get('label_free') is not True:
            raise ValueError(f'candidate dump is not label_free: {path}')
        person_id = _identity(payload.get('person_id'), 'person')
        question_id = _identity(payload.get('question_id'), 'question')
        if path.parent.name != person_id or path.stem != question_id:
            raise ValueError(f'candidate dump path identity mismatch: {path}')
        if payload.get('candidate_width') != CANDIDATE_WIDTH:
            raise ValueError(f'candidate width must be {CANDIDATE_WIDTH}: {path}')
        demands = payload.get('demands')
        if not isinstance(demands, list):
            raise ValueError(f'candidate dump demands must be a list: {path}')
        if payload.get('demand_count') != len(demands):
            raise ValueError(f'candidate dump demand count mismatch: {path}')
        demand_ids = set()
        for demand in demands:
            if not isinstance(demand, dict):
                raise ValueError(f'candidate demand must be an object: {path}')
            demand_id = _identity(demand.get('demand_id'), 'demand')
            if demand_id in demand_ids:
                raise ValueError(f'duplicate candidate demand: {path}: {demand_id}')
            demand_ids.add(demand_id)
            candidates = demand.get('candidates')
            if (not isinstance(candidates, list)
                    or demand.get('candidate_count') != CANDIDATE_WIDTH
                    or len(candidates) != CANDIDATE_WIDTH):
                raise ValueError(
                    f'every demand must contain {CANDIDATE_WIDTH} candidates: '
                    f'{path}: {demand_id}')
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(f'candidate must be an object: {path}')
                if candidate.get('modality') not in {'text', 'image'}:
                    raise ValueError(f'candidate modality invalid: {path}')
                _finite_number(candidate.get('rerank_score'), 'rerank_score')
                _string_list(candidate.get('object_ids'), 'candidate object_ids')
                _string_list(candidate.get('session_ids'), 'candidate session_ids')
        identity = (person_id, question_id)
        if identity in questions:
            raise ValueError(f'duplicate candidate question: {identity}')
        questions[identity] = payload
        users.add(person_id)
    if not any(demand['candidates']
               for payload in questions.values()
               for demand in payload['demands']):
        raise ValueError(f'candidate dumps contain no candidates under {root}')
    return questions, paths, sorted(users)


def load_source_objects(labels_path, users, index_root=None):
    """Resolve the T1.3 source indexes needed for image-sha256 matching."""
    source_root, paths = resolve_source_indexes(
        labels_path, index_root, users)
    sources = {}
    for user_id, path in sorted(paths.items()):
        payload = json.loads(path.read_text())
        if (payload.get('schema') != 'source-object-container-v2'
                or str(payload.get('person_id')) != user_id):
            raise ValueError(f'source index identity mismatch: {path}')
        objects = payload.get('objects')
        if not isinstance(objects, dict):
            raise ValueError(f'source objects missing: {path}')
        sources[user_id] = objects
    return source_root, sources


def _label_evidence(label, image_ref_map):
    sessions = set(_string_list(
        label.get('source_session_ids'), 'source_session_ids'))
    refs = label.get('evidence_image_refs')
    if (not isinstance(refs, list)
            or any(not isinstance(ref, dict) for ref in refs)):
        raise ValueError('evidence_image_refs must be a list of objects')
    images = set()
    unmapped = []
    for ref in refs:
        matches = {image_ref_map[key] for key in _image_ref_keys(ref)
                   if key in image_ref_map}
        if len(matches) > 1:
            raise ValueError(f'label image reference is ambiguous: {ref}')
        if matches:
            images.update(matches)
        else:
            unmapped.append(ref)
    return sessions, images, unmapped


def _candidate_image_shas(candidate, objects):
    shas = set()
    for object_id in candidate['object_ids']:
        obj = objects.get(object_id)
        if not isinstance(obj, dict):
            raise ValueError(f'candidate references unknown source object: {object_id}')
        if obj.get('kind') == 'image':
            shas.add(_sha256(obj.get('source_sha256'), 'source_sha256'))
        elif obj.get('kind') != 'text':
            raise ValueError(f'unknown source object kind: {obj.get("kind")}')
    if candidate['modality'] == 'image' and not shas:
        raise ValueError('image candidate has no image source object')
    return shas


def _question_rows(payload, label, objects, image_ref_map):
    sessions, images, unmapped_images = _label_evidence(label, image_ref_map)
    rows = []
    for demand in payload['demands']:
        for candidate in demand['candidates']:
            candidate_sessions = set(candidate['session_ids'])
            candidate_images = _candidate_image_shas(candidate, objects)
            positive = bool(candidate_sessions & sessions
                            or candidate_images & images)
            rows.append({
                'modality': candidate['modality'],
                'rerank_score': float(candidate['rerank_score']),
                'session_ids': candidate_sessions,
                'image_sha256': candidate_images,
                'positive': positive,
            })
    return rows, sessions, images, unmapped_images


def _question_coverage_score(rows, sessions, images, unmapped_images):
    units = [('session', value) for value in sorted(sessions)]
    units.extend(('image', value) for value in sorted(images))
    if unmapped_images:
        return -math.inf
    if not units:
        return None
    best = {}
    for row in rows:
        for kind, value in units:
            matches = (value in row['session_ids'] if kind == 'session'
                       else value in row['image_sha256'])
            if matches:
                key = (kind, value)
                best[key] = max(best.get(key, -math.inf), row['rerank_score'])
    if len(best) != len(units):
        return -math.inf
    return min(best.values())


def conformal_tau(scores, alpha):
    """Return the T3.2 conformal threshold, using ``-inf`` for an empty rank."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError('alpha must be numeric')
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError('alpha must lie strictly between zero and one')
    ordered = sorted(float(value) for value in scores)
    index = math.floor(float(alpha) * (len(ordered) + 1))
    if index < 1:
        return -math.inf
    return ordered[index - 1]


def _fit_isotonic(rows):
    if not rows:
        return None
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    model = IsotonicRegression(increasing=True, out_of_bounds='clip')
    model.fit(x, y)
    probabilities = np.clip(
        np.asarray(model.predict(x), dtype=float), *PROBABILITY_CLIP)
    fitted_y = np.clip(
        np.asarray(model.y_thresholds_, dtype=float), *PROBABILITY_CLIP)
    return {
        'method': 'isotonic_regression',
        'n': len(rows),
        'positive_count': int(y.sum()),
        'negative_count': len(rows) - int(y.sum()),
        'breakpoints': {
            'rerank_score': [float(value) for value in model.X_thresholds_],
            'calibrated_probability': [float(value) for value in fitted_y],
        },
        'probability_clip': list(PROBABILITY_CLIP),
        '_probabilities': probabilities,
    }


def _reliability_bins(rows, probabilities):
    bins = []
    total = len(rows)
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        members = [
            (row, probability) for row, probability in zip(rows, probabilities)
            if (index == 9 and lower <= probability <= upper)
            or (index < 9 and lower <= probability < upper)
        ]
        count = len(members)
        if count:
            confidence = sum(probability for _, probability in members) / count
            accuracy = sum(row[1] for row, _ in members) / count
            gap = abs(accuracy - confidence)
            ece += count / total * gap
        else:
            confidence = None
            accuracy = None
            gap = None
        bins.append({
            'lower': lower,
            'upper': upper,
            'count': count,
            'mean_confidence': confidence,
            'empirical_accuracy': accuracy,
            'absolute_gap': gap,
        })
    return bins, ece


def _payload_sha256(payload):
    body = {key: value for key, value in payload.items() if key != 'sha256'}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validate_calibrator(payload):
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA:
        raise ValueError('calibrator schema mismatch')
    declared = _sha256(payload.get('sha256'), 'calibrator sha256')
    if declared != _payload_sha256(payload):
        raise ValueError('calibrator sha256 mismatch')
    return payload


def build_calibrator(dump_root, labels_path, index_root=None):
    questions, dump_paths, users = load_candidate_dumps(dump_root)
    labels = load_labels(labels_path)
    source_root, sources = load_source_objects(
        labels_path, users, index_root=index_root)
    image_maps = {
        user_id: build_image_ref_map(objects)
        for user_id, objects in sources.items()
    }

    modality_rows = {'text': [], 'image': []}
    coverage_scores = []
    questions_without_evidence = 0
    unmapped_image_ref_count = 0
    for identity, payload in sorted(questions.items()):
        person_id, question_id = identity
        label = labels.get(identity)
        if label is None:
            raise ValueError(f'candidate question has no label: {identity}')
        rows, sessions, images, unmapped_images = _question_rows(
            payload, label, sources[person_id], image_maps[person_id])
        for row in rows:
            modality_rows[row['modality']].append(
                (row['rerank_score'], int(row['positive'])))
        score = _question_coverage_score(
            rows, sessions, images, unmapped_images)
        if score is None:
            questions_without_evidence += 1
        else:
            coverage_scores.append(score)
        unmapped_image_ref_count += len(unmapped_images)

    calibration = {}
    diagnostics_by_modality = {}
    for modality in ('text', 'image'):
        rows = modality_rows[modality]
        fitted = _fit_isotonic(rows)
        probabilities = fitted.pop('_probabilities') if fitted else np.asarray([])
        bins, ece = _reliability_bins(rows, probabilities) if rows else ([], None)
        calibration[modality] = fitted or {
            'method': 'isotonic_regression',
            'n': 0,
            'positive_count': 0,
            'negative_count': 0,
            'breakpoints': {
                'rerank_score': [],
                'calibrated_probability': [],
            },
            'probability_clip': list(PROBABILITY_CLIP),
        }
        diagnostics_by_modality[modality] = {
            'n': len(rows),
            'positive_count': sum(label for _, label in rows),
            'reliability_bins': bins,
            'ece': ece,
        }

    tau = {str(alpha): conformal_tau(coverage_scores, alpha)
           for alpha in ALPHAS}
    payload = {
        'schema': SCHEMA,
        'split': 'development',
        'users': users,
        'inputs': {
            'dump_root': str(Path(dump_root).resolve()),
            'candidate_dump_count': len(dump_paths),
            'labels': {
                'path': str(Path(labels_path).resolve()),
                'sha256': digest(labels_path),
            },
            'source_root': str(Path(source_root).resolve()),
        },
        'candidate_width': CANDIDATE_WIDTH,
        'calibration': calibration,
        'alphas': list(ALPHAS),
        'tau': {key: (None if math.isinf(value) else value)
                for key, value in tau.items()},
        'tau_negative_infinity': [
            str(alpha) for alpha in ALPHAS if math.isinf(tau[str(alpha)])],
        'tau_encoding': 'JSON null denotes negative infinity',
        'diagnostics': {
            'n': len(coverage_scores),
            'questions_total': len(questions),
            'questions_with_evidence_units': len(coverage_scores),
            'questions_without_evidence_units': questions_without_evidence,
            'g_negative_infinity_count': sum(
                math.isinf(value) and value < 0 for value in coverage_scores),
            'g_negative_infinity_ratio': (
                sum(math.isinf(value) and value < 0
                    for value in coverage_scores) / len(coverage_scores)
                if coverage_scores else None),
            'unmapped_evidence_image_refs': unmapped_image_ref_count,
            'modalities': diagnostics_by_modality,
        },
        'coverage_note': COVERAGE_NOTE,
    }
    payload['sha256'] = _payload_sha256(payload)
    return validate_calibrator(payload)


def write_calibrator(payload, output):
    validate_calibrator(payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True,
        allow_nan=False) + '\n')
    temporary.replace(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index-root', type=Path)
    args = parser.parse_args(argv)
    calibrator = build_calibrator(
        args.dump, args.labels, index_root=args.index_root)
    output = write_calibrator(calibrator, args.out)
    print(output)
    print(json.dumps({
        'users': calibrator['users'],
        'tau': calibrator['tau'],
        'diagnostics': {
            'n': calibrator['diagnostics']['n'],
            'g_negative_infinity_ratio': (
                calibrator['diagnostics']['g_negative_infinity_ratio']),
            'text_n': calibrator['diagnostics']['modalities']['text']['n'],
            'image_n': calibrator['diagnostics']['modalities']['image']['n'],
        },
        'sha256': calibrator['sha256'],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
