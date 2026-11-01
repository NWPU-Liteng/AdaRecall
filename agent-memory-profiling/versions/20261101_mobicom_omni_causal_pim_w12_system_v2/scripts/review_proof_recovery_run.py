"""Post-run scientific audit; never feeds labels back into candidate selection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def latency(values):
    ordered = sorted(values)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError('nonempty finite nonnegative measured samples required')
    return {'sum_seconds': sum(ordered), 'mean_seconds': statistics.mean(ordered),
            'p50_nearest_rank_seconds': ordered[math.ceil(.50 * len(ordered)) - 1],
            'p95_nearest_rank_seconds': ordered[math.ceil(.95 * len(ordered)) - 1]}


def mean_defined(values):
    defined = [value for value in values if value is not None]
    return {'defined_requests': len(defined), 'mean': statistics.mean(defined) if defined else None}


def recall(expected, actual):
    return len(expected & actual) / len(expected) if expected else None


def bind_judgments(rows):
    bound = {}
    for row in rows:
        key = (str(Path(row['run']).resolve()), row['question_id'])
        if key in bound:
            raise ValueError('duplicate Judge request; repeated trials must not overwrite one another')
        if (type(row['parse_valid']) is not bool
                or row['parse_valid'] != (row['judge_label'] in {'CORRECT', 'WRONG'})):
            raise ValueError('Judge label/validity contract differs')
        bound[key] = row
    return bound


def quality_summary(diagnostics):
    correct = sum(row['correct'] for row in diagnostics)
    return {'requests': len(diagnostics), 'correct': correct,
            'accuracy': correct / len(diagnostics) if diagnostics else None,
            'invalid_counted_wrong': sum(not row['judge_parse_valid'] for row in diagnostics)}


def mechanism_summary(rows):
    parsed_graphs = [(row['proof_plan']['query_parse'].get('parsed') or {}) for row in rows]
    usable_graphs = [graph if isinstance(graph, dict) else {} for graph in parsed_graphs]
    hard_counts = [sum(demand.get('hard') is True for demand in graph.get('demands', []))
                   for graph in usable_graphs]
    selected_record_counts = [len({identity for witness in row['proof_plan']['selected_witnesses']
        for identity in witness.get('retrieval_record_ids', [witness['base_record_id']])}) for row in rows]
    return {'requests': len(rows),
        'parse_status': dict(Counter(row['proof_plan']['query_parse']['status'] for row in rows)),
        'declared_hard_demand_count_histogram': dict(Counter(hard_counts)),
        'declared_relation_count_histogram': dict(Counter(len(graph.get('relations', [])) for graph in usable_graphs)),
        'multi_hard_demand_source_located_requests': sum(count > 1 and not row['proof_plan'].get('strict_proof_failure', False)
            for count, row in zip(hard_counts, rows)),
        'grounding_status': dict(Counter(row['proof_plan']['grounding']['status'] for row in rows)),
        'selected_witness_count_histogram': dict(Counter(len(row['proof_plan']['selected_witnesses']) for row in rows)),
        'selected_record_count_histogram': dict(Counter(selected_record_counts)),
        'single_record_fallback_requests': sum(count == 1 and row['proof_plan'].get('strict_proof_failure', False)
            for count, row in zip(selected_record_counts, rows)),
        'multi_record_source_located_nonfallback_requests': sum(count > 1 and not row['proof_plan'].get('strict_proof_failure', False)
            for count, row in zip(selected_record_counts, rows)),
        'selected_image_count_histogram': dict(Counter(len(row['images']) for row in rows)),
        'fallback_missing_domain_requests': sum(bool(row['proof_plan'].get('failed_strict_search', {}).get('missing_demands'))
            for row in rows),
        'final_search_proven_optimal_requests': sum(row['proof_plan']['optimal'] for row in rows),
        'final_search_not_proven_optimal_requests': sum(not row['proof_plan']['optimal'] for row in rows),
        'final_search_states': sum(row['proof_plan']['states'] for row in rows),
        'separator_dp_requests': sum(row['proof_plan']['structure']['separator_dp_used'] for row in rows),
        'candidate_identity_trace_complete': False,
        'semantic_entailment_proven': False,
        'boundary': 'source location and search optimality are not semantic sufficiency; soft-query single-record fallback is not a complementary proof'}


def summarize_locators(records, status):
    identities = [row['image_sha256'] for row in records]
    if len(identities) != len(set(identities)) or len(records) != status['images']:
        raise ValueError('visual extraction denominator/unique identity differs')
    unusable = sum(not row['parse_valid'] for row in records)
    if unusable != status['invalid_records']:
        raise ValueError('merged visual region accounting differs')
    return {'images': len(records),
        'raw_full_parse_invalid_before_placeholder_filter': sum(not row['original_parse_valid'] for row in records),
        'complete_region_records_after_placeholder_filter': len(records) - unusable,
        'no_complete_region_record_after_placeholder_filter': unusable,
        'non_eos_records': sum(not row['eos'] for row in records),
        'boundary': 'complete-region counts exclude subsequent recovery of complete region objects from partial JSON; neither parse validity nor recovery verifies visible facts'}


def review(root, prepared, archive_path):
    root, prepared = root.resolve(), prepared.resolve()
    cohort = json.loads((root / 'cohort.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    prepared_audit = json.loads((prepared / 'audit.json').read_text())
    if protocol['cohort_sha256'] != digest(root / 'cohort.json'):
        raise ValueError('registered cohort changed')
    if protocol['prepared_audit_sha256'] != digest(prepared / 'audit.json'):
        raise ValueError('registered prepared release changed')
    for name in ('labels.jsonl', 'manifests.jsonl', 'omni_graph.json', 'split.json'):
        if prepared_audit['artifacts_sha256'][name] != digest(prepared / name):
            raise ValueError('prepared evaluation file changed')
    runtime = root / 'runtime_mainline'
    if any(digest(runtime / name) != expected for name, expected in protocol['source_files_sha256'].items()):
        raise ValueError('frozen candidate source changed during execution')
    labels = {row['question_id']: row for row in map(json.loads, (prepared / 'labels.jsonl').read_text().splitlines())}
    manifests = {row['question_id']: row for row in map(json.loads, (prepared / 'manifests.jsonl').read_text().splitlines())}
    judge = json.loads((root / 'judge.json').read_text())
    if not judge.get('complete'):
        raise ValueError('whole-cohort fixed Judge must complete first')
    judgments = bind_judgments(judge['rows'])
    locator = json.loads((root / 'visual_merged/status.json').read_text())
    for name, field in (('records.jsonl', 'records_sha256'), ('image_hashes.json', 'image_hashes_sha256')):
        if digest(root / 'visual_merged' / name) != locator[field]:
            raise ValueError('sealed visual locator bytes changed')
    locator_records = list(map(json.loads, (root / 'visual_merged/records.jsonl').read_text().splitlines()))
    locator_summary = summarize_locators(locator_records, locator)
    locator_image_hashes = json.loads((root / 'visual_merged/image_hashes.json').read_text())
    if {row['image_sha256'] for row in locator_records} != set(locator_image_hashes.values()):
        raise ValueError('visual locators do not cover the registered content hashes')
    rows, diagnostics, source_sha = [], [], {}
    covered_judgments = set()
    image_hashes, sources, indices, indexed_visual_hashes = {}, {}, [], set()
    for user, identities in cohort['question_ids_by_user'].items():
        path = root / f'proof_user_{user}.json'
        run = json.loads(path.read_text())
        if (not run.get('complete') or [row['question_id'] for row in run['rows']] != identities
                or run['evaluation_label_files_read'] or run['validation_accessed'] or run['holdout_accessed']):
            raise ValueError('whole-cohort order and evaluation isolation required')
        run_digest = digest(path)
        if judge['provenance']['run_sha256'][str(path)] != run_digest:
            raise ValueError('Judge does not bind final saved run bytes')
        source_sha[str(path)] = run_digest
        index_root = root / f'proof_index_user_{user}'
        index = json.loads((index_root / 'index.json').read_text())
        if run['index_sha256'] != digest(index_root / 'index.json'):
            raise ValueError('generated run refers to another index')
        for name, field in (('records.json', 'records_sha256'), ('source.json', 'source_sha256'),
                            ('vectors.npy', 'vectors_sha256')):
            if digest(index_root / name) != index[field]:
                raise ValueError('index/source bytes changed')
        source = json.loads((index_root / 'source.json').read_text())
        sources[user] = source
        objects = source['objects']
        index_records = json.loads((index_root / 'records.json').read_text())
        visual_records = [record for record in index_records if record['kind'] == 'image_locator']
        indexed_visual_hashes.update(objects[record['image_object_id']]['source_sha256'] for record in visual_records)
        indices.append({'person_id': user, 'records': index['records'],
                        'image_locator_records': len(visual_records),
                        'image_locator_parse_status': dict(Counter(record['locator_parse_status'] for record in visual_records)),
                        'construction_seconds': index['construction_seconds'],
                        'storage_bytes': sum(item.stat().st_size for item in index_root.iterdir() if item.is_file())})
        for row in run['rows']:
            identity = row['question_id']
            if row['question_sha256'] != hashlib.sha256(manifests[identity]['question'].encode()).hexdigest():
                raise ValueError('question identity differs')
            if row['answer_sha256'] != hashlib.sha256(row['answer'].encode()).hexdigest():
                raise ValueError('answer identity differs')
            key = (str(path), identity)
            judgment = judgments[key]
            covered_judgments.add(key)
            if judgment['prediction_sha256'] != row['answer_sha256']:
                raise ValueError('judgment binds another prediction')
            plan = row['proof_plan']
            if row['evaluation_label_files_read']:
                raise ValueError('per-request evaluation isolation failed')
            selected_objects = set(plan['accepted_object_ids'])
            witness_union = {object_id for witness in plan['selected_witnesses'] for object_id in witness['object_ids']}
            if selected_objects != witness_union or selected_objects - objects.keys():
                raise ValueError('proof loses unique physical object identity')
            if plan['final_render']['input_tokens'] != row['input_tokens'] or row['input_tokens'] > protocol['input_tokens']:
                raise ValueError('actual processor budget differs from trace')
            if plan['final_render']['text_tokens'] > protocol['text_tokens'] or len(row['images']) > protocol['max_images']:
                raise ValueError('registered text/image resource limit exceeded')
            if not row['answer_token_ids'] or len(row['answer_token_ids']) > protocol['answer_max_new_tokens']:
                raise ValueError('empty answer or generation budget violated')
            if row['ttft_seconds'] > row['service_seconds'] or row['ttft_seconds'] < row['prefill_seconds']:
                raise ValueError('full-service TTFT bracket is inconsistent')
            if row['backend_plan']['chosen_backend'] != 'host' or row['backend_plan']['physical_execution']:
                raise ValueError('unqualified PIM claim entered serving output')
            selected_members, selected_sessions, selected_hashes = set(), set(), set()
            source_hints = {identity: source for witness in plan['selected_witnesses']
                            for identity, source in witness.get('source_hints', {}).items()}
            provenance_header_mismatches = []
            for object_id in selected_objects:
                obj = objects[object_id]
                if obj['kind'] == 'text':
                    if hashlib.sha256(obj['raw_text'].encode()).hexdigest() != obj['source_sha256']:
                        raise ValueError('raw text object hash differs')
                    selected_sessions.add(obj['session_id'])
            for ref in row['images']:
                member = ref.get('resolved', ref['normalized'])
                selected_members.add(member)
                selected_sessions.add(ref['source_session_id'])
                expected_hash = objects[ref['object_id']]['source_sha256']
                if expected_hash != ref['image_sha256']:
                    raise ValueError('image allocation hash differs')
                image_hashes[member] = expected_hash
                selected_hashes.add(expected_hash)
                hint = source_hints.get(ref['object_id'])
                if hint and hint['atom_id'] != ref['source_atom_id']:
                    provenance_header_mismatches.append({'object_id': ref['object_id'],
                        'proof_source_atom': hint['atom_id'], 'reader_header_atom': ref['source_atom_id']})
            expected_images = {ref.get('resolved', ref['normalized']) for ref in labels[identity]['evidence_image_refs']}
            expected_image_hashes = {locator_image_hashes[member] for member in expected_images if member in locator_image_hashes}
            expected_sessions = set(labels[identity]['source_session_ids'])
            correct = bool(judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT')
            diagnostics.append({'person_id': user, 'question_id': identity, 'correct': correct,
                'judge_parse_valid': judgment['parse_valid'], 'stop_reason': row['stop_reason'],
                'strict_proof_used': not plan.get('strict_proof_failure', False),
                'selected_image_reference_recall': recall(expected_images, selected_members),
                'selected_observable_image_content_recall': recall(expected_image_hashes, selected_hashes),
                'proof_image_source_header_mismatches': provenance_header_mismatches,
                'selected_source_session_recall': recall(expected_sessions, selected_sessions),
                'selected_object_count': len(selected_objects), 'selected_image_count': len(row['images']),
                'annotated_image_references': len(expected_images),
                'selected_annotated_image_references': len(expected_images & selected_members),
                'annotated_image_references_without_observable_hash': len(expected_images - locator_image_hashes.keys()),
                'candidate_image_reference_recall': None,
                'candidate_coverage_missing_reason': 'online candidate identities/query vectors not fully persisted; no posthoc retrieval replay counted as same-run evidence'})
            rows.append(row)
    if set(judgments) != covered_judgments:
        raise ValueError('Judge denominator has missing/extra candidate trials')
    with zipfile.ZipFile(archive_path) as archive:
        for member, expected_hash in image_hashes.items():
            if hashlib.sha256(archive.read(member)).hexdigest() != expected_hash:
                raise ValueError('selected original image archive bytes differ')
    stages = {}
    for phase in ('query_parse_and_retrieval', 'source_grounding', 'proof_materialize', 'reader_prefill', 'reader_decode'):
        stages[phase] = sum(next(event['seconds'] for event in row['execution_events'] if event['phase'] == phase)
                            for row in rows)
    residuals = [row['service_seconds'] - sum(event['seconds'] for event in row['execution_events']) for row in rows]
    if min(residuals) < -1e-6:
        raise ValueError('overlapping event timers cannot be added as disjoint service work')
    extraction = [shard['status'] for shard in locator['shards']]
    correct = sum(row['correct'] for row in diagnostics)
    grouped_quality = {'per_user': {user: quality_summary([row for row in diagnostics if row['person_id'] == user])
                                   for user in cohort['question_ids_by_user']},
        'source_located_nonfallback': quality_summary([row for row in diagnostics if row['strict_proof_used']]),
        'retrieval_fallback': quality_summary([row for row in diagnostics if not row['strict_proof_used']])}
    annotated_images = sum(row['annotated_image_references'] for row in diagnostics)
    selected_annotated_images = sum(row['selected_annotated_image_references'] for row in diagnostics)
    return {'schema': 'proof-recovery-scientific-review-v2', 'complete': True,
        'requests': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
        'quality_by_execution_path': grouped_quality,
        'method_mechanism': mechanism_summary(rows),
        'invalid_judgments_counted_wrong': sum(not row['judge_parse_valid'] for row in diagnostics),
        'strict_eos_correct': sum(row['correct'] and row['stop_reason'] == 'eos' for row in diagnostics),
        'grounding_rejection_reasons': dict(Counter(item['reason'] for row in rows
            for item in row['proof_plan']['grounding']['rejections'])),
        'strict_proof_requests': sum(row['strict_proof_used'] for row in diagnostics),
        'fallback_requests': sum(not row['strict_proof_used'] for row in diagnostics),
        'selected_image_occurrences': sum(len(row['images']) for row in rows),
        'selected_unique_image_bytes_verified': len(image_hashes),
        'proof_image_source_header_mismatches': sum(len(row['proof_image_source_header_mismatches']) for row in diagnostics),
        'selected_image_reference_recall': mean_defined([row['selected_image_reference_recall'] for row in diagnostics]),
        'selected_image_reference_micro_recall': {'annotated_references': annotated_images,
            'selected_annotated_references': selected_annotated_images,
            'recall': selected_annotated_images / annotated_images if annotated_images else None},
        'selected_observable_image_content_recall': mean_defined([row['selected_observable_image_content_recall'] for row in diagnostics]),
        'selected_source_session_recall': mean_defined([row['selected_source_session_recall'] for row in diagnostics]),
        'service_latency': latency([row['service_seconds'] for row in rows]),
        'ttft_latency': latency([row['ttft_seconds'] for row in rows]),
        'same_run_disjoint_stages_seconds': stages, 'unassigned_service_bracket_seconds': sum(residuals),
        'output_tokens': sum(len(row['answer_token_ids']) for row in rows),
        'input_tokens': {'min': min(row['input_tokens'] for row in rows),
            'mean': statistics.mean(row['input_tokens'] for row in rows),
            'max': max(row['input_tokens'] for row in rows),
            'mean_fraction_of_registered_cap': statistics.mean(row['input_tokens'] for row in rows) / protocol['input_tokens']},
        'truncated_requests': [row['question_id'] for row in rows if row['stop_reason'] != 'eos'],
        'index_construction': indices,
        'offline_visual_extraction': {**locator_summary,
            'unique_indexed_image_contents_after_partial_region_recovery': len(indexed_visual_hashes),
            'sum_worker_elapsed_seconds': sum(item['elapsed_seconds'] for item in extraction),
            'sum_batch_generation_seconds': sum(batch['seconds'] for item in extraction for batch in item['batches']),
            'max_worker_elapsed_seconds_not_final_pipeline_makespan': max(item['elapsed_seconds'] for item in extraction)},
        'provenance': {'runs_sha256': source_sha, 'judge_sha256': digest(root / 'judge.json'),
                       'protocol_sha256': digest(root / 'protocol.json'), 'code_file_count': len(protocol['source_files_sha256']),
                       'postrun_review_script_sha256': digest(Path(__file__))},
        'main_experiment_closed': False, 'independent_validation': False,
        'new_baseline_or_ablation': False, 'causal_e2e_speedup': None, 'target_pim_e2e_speedup': None,
        'implementation_gaps': ['no separator DP path; augmented exact-render factor handled by bounded join search',
            'source-located span/region does not prove semantic entailment or factual sufficiency',
            'lifecycle exports current-request source logical bytes, not a measured full tensor pebbling optimizer',
            'backend is Host only; no calibrated full-system capacity/cost or joint residency/lifetime optimization',
            'candidate identities and complete per-query operator trace are not fully persisted',
            'no matched new baseline, independent users or isolated latency repeat; historical 83/192 is only context'],
        'diagnostics': diagnostics,
        'boundary': 'post-run evaluation only, with full denominator; coverage is provenance annotation coverage, not image grounding; no mobile/PIM or novelty claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh scientific review output required')
    result = review(args.root, args.prepared, args.image_archive)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('complete', 'requests', 'correct', 'strict_proof_requests',
        'fallback_requests', 'selected_image_occurrences', 'main_experiment_closed')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
