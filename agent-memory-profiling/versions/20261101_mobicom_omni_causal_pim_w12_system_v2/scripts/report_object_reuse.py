"""Label-free object reuse analysis for a completed proof-mainline run directory."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import sys


SCHEMA = 'object-reuse-analysis-v1'
DEFAULT_CAPACITIES = (8, 16, 32, 64, 128, 256, 512)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{path} must be a positive integer')
    return value


def _nonnegative_integer(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{path} must be a non-negative integer')
    return value


def load_image_costs(paths):
    costs = {}
    sources = {}
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f'image costs must be a JSON object: {path}')
        for image_sha256, value in payload.items():
            if not isinstance(image_sha256, str) or len(image_sha256) != 64:
                raise ValueError(f'invalid image identity in {path}: {image_sha256!r}')
            value = _positive_integer(value, f'{path}:{image_sha256}')
            if image_sha256 in costs and costs[image_sha256] != value:
                raise ValueError(f'image token costs conflict for {image_sha256}')
            costs[image_sha256] = value
        sources[str(path.resolve())] = file_digest(path)
    return costs, sources


def load_run_shards(run_dir):
    run_dir = Path(run_dir)
    paths = [path for path in sorted(run_dir.glob('user_*_shard_*.json'))
             if not path.name.endswith('.partial.json')]
    if not paths:
        raise FileNotFoundError(f'no complete user shard outputs in {run_dir}')
    users = {}
    reader_models = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get('complete') is not True:
            raise ValueError(f'incomplete user shard output: {path}')
        if payload.get('capacity_capture_only') is True:
            raise ValueError(f'capture-only run has no selected objects: {path}')
        person_id = payload.get('person_id')
        question_ids = payload.get('question_ids')
        rows = payload.get('rows')
        shard_index = payload.get('shard_index')
        shard_count = payload.get('shard_count')
        reader_model = payload.get('reader_model')
        if (isinstance(person_id, bool) or not isinstance(person_id, (str, int))
                or not str(person_id) or not isinstance(reader_model, str)
                or not reader_model):
            raise ValueError(f'run user or Reader identity missing: {path}')
        if (not isinstance(question_ids, list) or not question_ids
                or any(not isinstance(value, str) or not value for value in question_ids)
                or len(question_ids) != len(set(question_ids))):
            raise ValueError(f'run question order is invalid: {path}')
        if not isinstance(rows, list) or len(rows) != len(question_ids):
            raise ValueError(f'run rows differ from question order: {path}')
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f'run row must be an object: {path}')
        if [row.get('question_id') for row in rows] != question_ids:
            raise ValueError(f'run row order differs from question_ids: {path}')
        shard_index = _nonnegative_integer(shard_index, f'{path}:shard_index')
        shard_count = _positive_integer(shard_count, f'{path}:shard_count')
        if shard_index >= shard_count:
            raise ValueError(f'shard index is outside shard count: {path}')
        person_id = str(person_id)
        if any(str(row.get('person_id')) != person_id for row in rows):
            raise ValueError(f'row user differs from shard user: {path}')
        request_indexes = [row.get('request_index') for row in rows]
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in request_indexes)
                or len(request_indexes) != len(set(request_indexes))):
            raise ValueError(f'run request indexes are invalid: {path}')
        entry = users.setdefault(person_id, {
            'shard_count': shard_count, 'shards': {}, 'files': []})
        if entry['shard_count'] != shard_count:
            raise ValueError(f'inconsistent shard count for user {person_id}')
        if shard_index in entry['shards']:
            raise ValueError(f'duplicate shard output for user {person_id}: {shard_index}')
        entry['shards'][shard_index] = {'question_ids': question_ids, 'rows': rows}
        entry['files'].append(str(path.resolve()))
        reader_models.add(reader_model)
    if len(reader_models) != 1:
        raise ValueError(f'one Reader model identity required: {sorted(reader_models)}')

    ordered_users = {}
    for person_id in sorted(users, key=lambda value: (int(value) if value.isdigit() else value)):
        entry = users[person_id]
        shard_count = entry['shard_count']
        if set(entry['shards']) != set(range(shard_count)):
            raise ValueError(f'incomplete shard set for user {person_id}')
        ids_by_shard = [entry['shards'][index]['question_ids']
                        for index in range(shard_count)]
        rows_by_question = {}
        for index in range(shard_count):
            shard = entry['shards'][index]
            for question_id, row in zip(shard['question_ids'], shard['rows']):
                if question_id in rows_by_question:
                    raise ValueError(f'duplicate question in user shards: {question_id}')
                rows_by_question[question_id] = row
        order = []
        max_local = max(len(values) for values in ids_by_shard)
        for local_index in range(max_local):
            for shard_index in range(shard_count):
                if local_index < len(ids_by_shard[shard_index]):
                    order.append(ids_by_shard[shard_index][local_index])
        if len(order) != len(rows_by_question) or len(order) != len(set(order)):
            raise ValueError(f'shard question order is inconsistent for user {person_id}')
        ordered_users[person_id] = {
            'questions': [rows_by_question[question_id] for question_id in order],
            'question_ids': order,
            'files': sorted(entry['files']),
        }
    return ordered_users, next(iter(reader_models)), paths


def extract_text_costs(row, tokenizer):
    plan = row.get('proof_plan')
    objects = row.get('materialized_object_ids')
    if not isinstance(objects, list):
        raise ValueError(f"materialized_object_ids missing for {row.get('question_id')}")
    if plan is None:
        if not objects:
            return {}
        raise ValueError(f"proof_plan missing for {row.get('question_id')}")
    if not isinstance(plan, dict):
        raise ValueError(f"proof_plan must be an object for {row.get('question_id')}")
    atom_ids = plan.get('accepted_atom_ids')
    assessment = plan.get('assessment')
    if not isinstance(atom_ids, list) or any(not isinstance(value, str) for value in atom_ids):
        raise ValueError(f"accepted_atom_ids missing for {row.get('question_id')}")
    if not isinstance(assessment, dict) or not isinstance(assessment.get('text'), str):
        raise ValueError(f"assessment text missing for {row.get('question_id')}")
    text_objects = [value for value in objects if value.startswith('text:')]
    expected_objects = {'text:' + atom_id for atom_id in atom_ids}
    if (len(atom_ids) != len(set(atom_ids))
            or len(text_objects) != len(set(text_objects))
            or set(text_objects) != expected_objects):
        raise ValueError(f"text object identities differ for {row.get('question_id')}")
    if not atom_ids:
        if assessment.get('text_tokens') != 0 or assessment['text']:
            raise ValueError(f"empty atom list has rendered text for {row.get('question_id')}")
        return {}

    text = assessment['text']
    located = []
    for atom_id in atom_ids:
        if ':' not in atom_id:
            raise ValueError(f'atom identity lacks session prefix: {atom_id}')
        session_id = atom_id.split(':', 1)[0]
        marker = f'[{session_id}][{atom_id}]['
        first = text.find(marker)
        if first < 0 or text.find(marker, first + 1) >= 0:
            raise ValueError(f'text marker is absent or ambiguous: {atom_id}')
        located.append((first, atom_id))
    located.sort()
    costs = {}
    for index, (start, atom_id) in enumerate(located):
        end = located[index + 1][0] if index + 1 < len(located) else len(text)
        cost = len(tokenizer.encode(text[start:end], add_special_tokens=False))
        if cost <= 0:
            raise ValueError(f'text object has no tokens: {atom_id}')
        costs['text:' + atom_id] = cost
    expected_tokens = assessment.get('text_tokens')
    if (isinstance(expected_tokens, bool) or not isinstance(expected_tokens, int)
            or sum(costs.values()) != expected_tokens):
        raise ValueError(f"text token accounting differs for {row.get('question_id')}")
    return costs


def extract_image_costs(row, exact_costs):
    plan = row.get('proof_plan')
    objects = row.get('materialized_object_ids')
    if not isinstance(plan, dict) or not isinstance(objects, list):
        raise ValueError(f"image accounting inputs missing for {row.get('question_id')}")
    assessment = plan.get('assessment')
    if not isinstance(assessment, dict):
        raise ValueError(f"assessment missing for {row.get('question_id')}")
    visual_tokens = assessment.get('visual_tokens')
    if isinstance(visual_tokens, bool) or not isinstance(visual_tokens, int) or visual_tokens < 0:
        raise ValueError(f"visual_tokens invalid for {row.get('question_id')}")
    image_objects = [value for value in objects if value.startswith('image:')]
    if len(image_objects) != len(set(image_objects)):
        raise ValueError(f"duplicate image object for {row.get('question_id')}")
    if not image_objects:
        if visual_tokens != 0:
            raise ValueError(f"visual tokens without images for {row.get('question_id')}")
        return {}, 'none', []
    identities = [value.split(':', 1)[1] for value in image_objects]
    missing = [identity for identity in identities if identity not in exact_costs]
    if not missing:
        costs = {object_id: exact_costs[identity]
                 for object_id, identity in zip(image_objects, identities)}
        if sum(costs.values()) != visual_tokens:
            raise ValueError(f"image cost sidecar differs from run for {row.get('question_id')}")
        return costs, 'image_costs_json', []
    if visual_tokens < len(image_objects):
        raise ValueError(f"too few visual tokens to allocate for {row.get('question_id')}")
    quotient, remainder = divmod(visual_tokens, len(image_objects))
    costs = {object_id: quotient + (1 if index < remainder else 0)
             for index, object_id in enumerate(image_objects)}
    return costs, 'equal_share_estimate', missing


def analyze_question(row, tokenizer, exact_image_costs):
    question_id = row.get('question_id')
    if not isinstance(question_id, str) or not question_id:
        raise ValueError('question identity missing')
    objects = row.get('materialized_object_ids')
    if not isinstance(objects, list) or any(not isinstance(value, str) for value in objects):
        raise ValueError(f'materialized object list invalid for {question_id}')
    if len(objects) != len(set(objects)):
        raise ValueError(f'duplicate materialized object for {question_id}')
    if not objects:
        if row.get('status') == 'no_visible_history':
            return {'question_id': question_id, 'question_index': row.get('request_index'),
                    'object_count': 0, 'object_tokens': 0, 'text_tokens': 0,
                    'visual_tokens': 0, 'image_token_cost_source': 'none',
                    'missing_image_cost_count': 0, 'objects': []}
        raise ValueError(f'question has no materialized objects: {question_id}')
    text_costs = extract_text_costs(row, tokenizer)
    image_costs, image_source, missing_images = extract_image_costs(
        row, exact_image_costs)
    unknown = sorted(set(objects) - set(text_costs) - set(image_costs))
    extra = sorted((set(text_costs) | set(image_costs)) - set(objects))
    if unknown or extra:
        raise ValueError(f'object cost mapping differs for {question_id}: '
                         f'unknown={unknown[:3]} extra={extra[:3]}')
    ordered = []
    for object_id in objects:
        kind = object_id.split(':', 1)[0]
        if kind not in {'text', 'image'}:
            raise ValueError(f'unknown object kind for {question_id}: {object_id}')
        token_cost = text_costs.get(object_id, image_costs.get(object_id))
        ordered.append({'object_id': object_id, 'kind': kind, 'tokens': token_cost})
    assessment = row['proof_plan']['assessment']
    return {
        'question_id': question_id,
        'question_index': row.get('request_index'),
        'object_count': len(ordered),
        'object_tokens': sum(item['tokens'] for item in ordered),
        'text_tokens': assessment.get('text_tokens'),
        'visual_tokens': assessment.get('visual_tokens'),
        'image_token_cost_source': image_source,
        'missing_image_cost_count': len(missing_images),
        'objects': ordered,
    }


def simulate_lru(questions, capacities):
    total_tokens = sum(item['tokens'] for question in questions
                       for item in question['objects'])
    results = []
    for capacity in capacities:
        cache = OrderedDict()
        saved_tokens = 0
        hits = misses = evictions = 0
        for question in questions:
            for item in question['objects']:
                object_id = item['object_id']
                if object_id in cache:
                    saved_tokens += item['tokens']
                    hits += 1
                    cache.move_to_end(object_id)
                else:
                    misses += 1
                    cache[object_id] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
                        evictions += 1
        results.append({
            'capacity_objects': capacity,
            'saved_tokens': saved_tokens,
            'saved_token_ratio': saved_tokens / total_tokens if total_tokens else None,
            'hit_occurrences': hits,
            'miss_occurrences': misses,
            'hit_rate': hits / (hits + misses) if hits + misses else None,
            'evictions': evictions,
            'final_resident_objects': len(cache),
            'unlimited_capacity_equivalent': capacity >= len({
                item['object_id'] for question in questions for item in question['objects']}),
        })
    return results


def analyze_user(person_id, questions, capacities):
    occurrences = []
    costs_by_object = defaultdict(list)
    kind_by_object = {}
    first_occurrence_tokens = 0
    seen = set()
    for question in questions:
        for item in question['objects']:
            occurrences.append(item)
            costs_by_object[item['object_id']].append(item['tokens'])
            previous_kind = kind_by_object.setdefault(item['object_id'], item['kind'])
            if previous_kind != item['kind']:
                raise ValueError(f'object kind changed: {item["object_id"]}')
            if item['object_id'] not in seen:
                seen.add(item['object_id'])
                first_occurrence_tokens += item['tokens']
    frequencies = Counter(item['object_id'] for item in occurrences)
    total_tokens = sum(item['tokens'] for item in occurrences)
    repeated = {object_id for object_id, frequency in frequencies.items() if frequency > 1}
    repeated_occurrences = sum(frequencies[object_id] for object_id in repeated)
    repeated_tokens = sum(item['tokens'] for item in occurrences
                          if item['object_id'] in repeated)
    frequency_histogram = Counter(frequencies.values())
    kind_occurrences = Counter(item['kind'] for item in occurrences)
    kind_distinct = Counter(kind_by_object.values())
    kind_frequency_histograms = {}
    for kind in sorted(kind_distinct):
        histogram = Counter(frequency for object_id, frequency in frequencies.items()
                            if kind_by_object[object_id] == kind)
        kind_frequency_histograms[kind] = {
            str(value): histogram[value] for value in sorted(histogram)}
    inconsistent_costs = [
        {'object_id': object_id, 'kind': kind_by_object[object_id],
         'token_cost_min': min(costs_by_object[object_id]),
         'token_cost_max': max(costs_by_object[object_id]),
         'occurrences': len(costs_by_object[object_id])}
        for object_id in sorted(costs_by_object)
        if len(set(costs_by_object[object_id])) > 1]
    reusable = []
    for object_id in repeated:
        object_occurrences = [item for item in occurrences
                              if item['object_id'] == object_id]
        object_total = sum(item['tokens'] for item in object_occurrences)
        reusable.append({
            'object_id': object_id,
            'kind': kind_by_object[object_id],
            'frequency': frequencies[object_id],
            'total_occurrence_tokens': object_total,
            'reusable_tokens': object_total - object_occurrences[0]['tokens'],
            'token_cost_min': min(costs_by_object[object_id]),
            'token_cost_max': max(costs_by_object[object_id]),
        })
    reusable.sort(key=lambda value: (-value['reusable_tokens'],
                                     -value['frequency'], value['object_id']))
    ideal_saved = total_tokens - first_occurrence_tokens
    return {
        'question_count': len(questions),
        'question_ids': [question['question_id'] for question in questions],
        'distinct_objects': len(frequencies),
        'object_occurrences': len(occurrences),
        'mean_frequency': len(occurrences) / len(frequencies) if frequencies else None,
        'max_frequency': max(frequencies.values()) if frequencies else 0,
        'frequency_histogram': {
            str(frequency): frequency_histogram[frequency]
            for frequency in sorted(frequency_histogram)},
        'kind_frequency_histograms': kind_frequency_histograms,
        'kind_occurrences': {kind: kind_occurrences[kind] for kind in sorted(kind_occurrences)},
        'kind_distinct_objects': {kind: kind_distinct[kind] for kind in sorted(kind_distinct)},
        'repeated_objects': len(repeated),
        'repeated_object_occurrences': repeated_occurrences,
        'repeated_object_token_volume': repeated_tokens,
        'duplicate_object_token_ratio': repeated_tokens / total_tokens if total_tokens else None,
        'selected_token_volume': total_tokens,
        'ideal_object_kv_reuse': {
            'saved_tokens_upper_bound': ideal_saved,
            'saved_token_ratio': ideal_saved / total_tokens if total_tokens else None,
            'cold_first_occurrence_tokens': first_occurrence_tokens,
        },
        'lru_object_capacity': simulate_lru(questions, capacities),
        'inconsistent_object_token_costs': inconsistent_costs,
        'top_reused_objects': reusable[:20],
        'questions': questions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tokenizer', type=Path,
                        help='Reader tokenizer path; defaults to run reader_model')
    parser.add_argument('--image-costs', type=Path, nargs='*', default=[],
                        help='proof_index_user_*/image_costs.json files')
    parser.add_argument('--capacities', type=int, nargs='+', default=list(DEFAULT_CAPACITIES))
    args = parser.parse_args()
    capacities = sorted(set(args.capacities))
    if not capacities or any(value <= 0 for value in capacities):
        raise ValueError('LRU capacities must be positive object counts')
    if args.output is not None and args.output.exists():
        raise ValueError(f'refusing to overwrite object reuse report: {args.output}')
    image_costs, image_cost_sources = load_image_costs(args.image_costs)
    users, reader_model, paths = load_run_shards(args.run_dir)
    tokenizer_path = args.tokenizer if args.tokenizer is not None else Path(reader_model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, padding_side='left')

    user_results = {}
    for person_id, entry in users.items():
        questions = [analyze_question(row, tokenizer, image_costs)
                     for row in entry['questions']]
        user_result = analyze_user(person_id, questions, capacities)
        user_result['run_files'] = entry['files']
        user_results[person_id] = user_result
    report = {
        'schema': SCHEMA,
        'analysis_estimate_only': True,
        'run_dir': str(args.run_dir.resolve()),
        'reader_model': reader_model,
        'tokenizer_model': str(tokenizer_path.resolve()),
        'capacities': capacities,
        'capacity_unit': 'distinct object identities',
        'question_order': ('reconstructed from shard_index/shard_count/question_ids; '
                           'row order within each shard is preserved'),
        'image_token_cost_source': ('image_costs_json when all images are present, '
                                    'otherwise equal-share estimate from assessment.visual_tokens'),
        'users': user_results,
        'run_files_sha256': {
            str(path.resolve()): file_digest(path) for path in paths},
        'image_costs_sha256': image_cost_sources,
        'script_sha256': file_digest(Path(__file__)),
        'evaluation_label_files_read': [],
        'boundary': ('analysis estimate over selected materialized objects; ideal and LRU '
                     'savings assume every selected object owns an independently reusable '
                     'KV span. Actual Reader KV layout, position/prefix constraints, decode '
                     'state, and image-feature reuse may change the realizable saving. '
                     'No labels or end-to-end latency are used.'),
    }
    if args.output is None:
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write('\n')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({
        'users': len(user_results),
        'questions': sum(value['question_count'] for value in user_results.values()),
        'distinct_objects': sum(value['distinct_objects'] for value in user_results.values()),
        'selected_token_volume': sum(value['selected_token_volume'] for value in user_results.values()),
        'output': str(args.output),
    }))


if __name__ == '__main__':
    main()
