"""Serving-artifact-only natural reuse opportunities and execution dependency DAG."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_omni_smoke_closure import digest, verify_source_snapshot


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def analyze_user(smoke):
    seen_batches, seen_images = {}, {}
    nodes, edges, rows = {}, set(), []
    image_occurrences = image_repeat_occurrences = batch_repeats = 0
    for index, row in enumerate(smoke['rows']):
        hashes = tuple(image['sha256'] for image in row['images'])
        if not hashes:
            raise ValueError('raw-image requests required')
        batch_identity = fingerprint(hashes)
        batch_node = 'vision-batch:' + batch_identity
        nodes[batch_node] = {'kind': 'ordered_vision_batch', 'image_sha256s': hashes,
                             'scope': 'same-user same-reader same-processor protocol'}
        repeated_images = []
        for image_hash in dict.fromkeys(hashes):
            raw_node = 'raw-image:' + image_hash
            pixel_node = 'processed-image-candidate:' + image_hash
            nodes[raw_node] = {'kind': 'immutable_raw_image', 'sha256': image_hash}
            nodes[pixel_node] = {'kind': 'candidate_processor_state',
                                 'qualification': 'requires actual pixel/grid equality'}
            edges.update(((raw_node, pixel_node), (pixel_node, batch_node)))
            if image_hash in seen_images:
                repeated_images.append({'sha256': image_hash,
                    'last_seen_request_distance': index - seen_images[image_hash]})
        repeated_count = sum(image_hash in seen_images for image_hash in hashes)
        is_batch_repeat = hashes in seen_batches
        batch_repeats += int(is_batch_repeat)
        image_occurrences += len(hashes)
        image_repeat_occurrences += repeated_count
        question = row['question_id']
        prefill_node, decode_node = 'prefill:' + question, 'decode:' + question
        nodes[prefill_node] = {'kind': 'request_context_dependent_prefill',
                               'question_id': question, 'reusable': False}
        nodes[decode_node] = {'kind': 'request_decode', 'question_id': question, 'reusable': False}
        edges.update(((batch_node, prefill_node), (prefill_node, decode_node)))
        rows.append({'question_id': question, 'ordered_batch_previously_seen': is_batch_repeat,
            'ordered_batch_distance': index - seen_batches[hashes] if is_batch_repeat else None,
            'previously_seen_image_occurrences': repeated_count,
            'image_count': len(hashes), 'image_reuse_distances': repeated_images,
            'new_batch_with_prior_image_overlap': not is_batch_repeat and repeated_count > 0})
        seen_batches[hashes] = index
        for image_hash in hashes:
            seen_images[image_hash] = index
    return {'requests': len(rows), 'ordered_batch_repeat_opportunities': batch_repeats,
        'ordered_batch_repeat_rate': batch_repeats / len(rows),
        'raw_image_occurrences': image_occurrences, 'unique_raw_images': len(seen_images),
        'previously_seen_image_occurrences': image_repeat_occurrences,
        'image_repeat_opportunity_rate': image_repeat_occurrences / image_occurrences,
        'new_batch_requests_with_prior_image_overlap': sum(
            row['new_batch_with_prior_image_overlap'] for row in rows),
        'embedding_store_observed': smoke.get('embedding_state_reuse'),
        'rows': rows, 'execution_dependency_dag': {'nodes': nodes, 'edges': sorted(edges)},
        'boundary': 'raw-image identity opportunities, not accepted exact feature-cache hits; no bounded-cache simulation, future reuse oracle, LLM KV reuse or quality labels; each user starts cold'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh natural-reuse diagnostic required')
    plan_path = args.run_root / 'cohort_plan.json'
    plan = json.loads(plan_path.read_text())
    users = []
    producer_hashes = {}
    for job in plan['jobs']:
        path = Path(job['output'])
        smoke = json.loads(path.read_text())
        registration = json.loads(path.with_suffix('.registration.json').read_text())
        if (not smoke.get('complete') or smoke.get('evaluation_label_files_read') != []
                or [row['question_id'] for row in smoke['rows']] != job['question_ids']
                or any(row['person_id'] != job['person_id'] for row in smoke['rows'])
                or not verify_source_snapshot(smoke, registration)
                or registration['source_files_sha256'] != smoke['source_files_sha256']
                or registration['question_ids'] != job['question_ids']
                or any(plan['source_files_sha256'].get(name) != value
                       for name, value in smoke['source_files_sha256'].items())):
            raise ValueError('frozen serving cohort source or request integrity failed')
        users.append({'person_id': job['person_id'], **analyze_user(smoke)})
        producer_hashes[str(path)] = digest(path)
    total_requests = sum(user['requests'] for user in users)
    total_images = sum(user['raw_image_occurrences'] for user in users)
    result = {'schema_version': 'omni-natural-reuse-dependency-diagnostic-v1',
        'cohort_plan_sha256': digest(plan_path), 'reader_sha256': producer_hashes,
        'script_sha256': digest(Path(__file__)), 'users': users,
        'total_requests': total_requests, 'total_image_occurrences': total_images,
        'ordered_batch_repeat_opportunities': sum(user['ordered_batch_repeat_opportunities'] for user in users),
        'raw_image_repeat_opportunities': sum(user['previously_seen_image_occurrences'] for user in users),
        'new_batch_requests_with_prior_image_overlap': sum(user['new_batch_requests_with_prior_image_overlap'] for user in users),
        'evaluation_label_files_read': [], 'feature_or_token_equivalence_qualified': False,
        'boundary': 'natural serving-only three-user streams; opportunities are not measured reuse or speedup; typed DAG is a structural representation, not a novel joint scheduler'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: result[name] for name in ('total_requests','total_image_occurrences',
        'ordered_batch_repeat_opportunities','raw_image_repeat_opportunities',
        'new_batch_requests_with_prior_image_overlap')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
