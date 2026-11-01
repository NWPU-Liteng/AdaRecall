"""Label-side MobileMem scoring; never imported by the generation runner."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


def load_pinned_protocol():
    upstream = ROOT / 'third_party/mobilemem_eval'
    manifest = json.loads((upstream / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((upstream / name).read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError(f'upstream evaluator changed: {name}')
    path = upstream / 'membase/evaluation/f1.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.FunctionDef))
                 or isinstance(node, ast.ImportFrom) and node.module == 'collections']
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    prompt_module = runpy.run_path(str(upstream / 'membase/inference_utils/prompts.py'))
    return namespace['_normalize_answer'], namespace['_token_f1'], prompt_module['get_prompt']('default-exact-match'), manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--answer-field', choices=('answer', 'first_answer'), default='answer')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh scoring output required')
    normalize_answer, token_f1, judge_template, upstream = load_pinned_protocol()
    labels = {item['question_id']: item for item in
              (json.loads(line) for line in (args.prepared / 'labels.jsonl').read_text().splitlines())}
    manifests = {item['question_id']: item for item in
                 (json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    graphs = json.loads((args.prepared / 'mobilemem_graph.json').read_text())
    event_timestamps = {event['event_id']: event['timestamp'] for graph in graphs for event in graph['nodes']}
    payload = json.loads(args.result.read_text())
    if not payload.get('complete'):
        raise ValueError('cannot score an incomplete run')
    expected_ids = payload['selected_question_ids']
    if (len(set(expected_ids)) != len(expected_ids)
            or [row['question_id'] for row in payload['rows']] != expected_ids):
        raise ValueError('missing, duplicate or reordered results')
    rows, judge_requests = [], []
    for row in payload['rows']:
        question_id = row['question_id']
        label = labels[question_id]
        manifest = manifests[question_id]
        if manifest['split'] != 'development':
            raise ValueError('sealed evaluation is not authorized by this development scorer')
        prediction = row.get(args.answer_field, '')
        succeeded = row.get('status') == 'ok'
        f1 = max((token_f1(prediction, answer) for answer in label['golden_answers']), default=0.0) if succeeded else 0.0
        exact = succeeded and any(normalize_answer(prediction) == normalize_answer(answer) for answer in label['golden_answers'])
        selected_ids = {atom['event_id'] for atom in row.get('selected_atoms', [])}
        source_ids = set(label['source_evidence_ids'])
        evidence_recall = len(selected_ids & source_ids) / len(source_ids) if source_ids else 0.0
        visible = all(event_timestamps.get(event_id, '9999') <= manifest['effective_timestamp']
                      for event_id in selected_ids)
        rows.append({'question_id': question_id, 'status': row.get('status'),
                     'question_type': label.get('question_type'), 'difficulty': label.get('difficulty'),
                     'token_f1': f1, 'exact_match': exact,
                     'source_event_hit_recall': evidence_recall, 'temporal_validity': visible,
                     'service_seconds': row.get('reader_service_seconds'),
                     'ttft_seconds': row.get('final_answer_ttft_service_seconds'),
                     'reused_tokens': row.get('prefix_state_reuse', {}).get('reused_tokens', 0),
                     'residual_proof_tokens': row.get('prefix_state_reuse', {}).get('residual_tokens'),
                     'selected_route': row.get('pim_retrieval', {}).get('selected_route'),
                     'pim_cycles': row.get('pim_retrieval', {}).get('pim_cycles'),
                     'host_cycles': row.get('pim_retrieval', {}).get('host_cycles')})
        golden = label['golden_answers']
        formatted_golden = golden[0] if len(golden) == 1 else f"[{', '.join(golden)}]"
        prompt = judge_template.substitute(question=manifest['question'],
                                           prediction=prediction, golden_answers=formatted_golden)
        judge_requests.append({'question_id': question_id, 'runtime_status': row.get('status'),
                               'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                               'prediction_sha256': hashlib.sha256(prediction.encode()).hexdigest()})
    valid = [row for row in rows if row['status'] == 'ok']
    def mean(key, subset=rows):
        values = [row[key] for row in subset if row.get(key) is not None]
        return sum(values) / len(values) if values else None
    def percentile(key, percentile_value):
        values = sorted(row[key] for row in valid if row.get(key) is not None)
        if not values:
            return None
        position = (len(values) - 1) * percentile_value / 100
        lower, upper = int(position), min(int(position) + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    summary = {'result': str(args.result), 'answer_field': args.answer_field,
               'rows': len(rows), 'runtime_failures': len(rows) - len(valid),
               'answer_success_rate': None, 'exact_match_rate': mean('exact_match'),
               'mean_token_f1': mean('token_f1'), 'mean_source_event_hit_recall': mean('source_event_hit_recall'),
               'temporal_validity_rate': mean('temporal_validity'), 'mean_reused_tokens': mean('reused_tokens'),
               'reuse_hit_rate': (sum(row['reused_tokens'] > 0 for row in valid) / len(valid) if valid else None),
               'mean_service_seconds': mean('service_seconds'), 'p50_service_seconds': percentile('service_seconds', 50),
               'p95_service_seconds': percentile('service_seconds', 95), 'mean_ttft_seconds': mean('ttft_seconds'),
               'p50_ttft_seconds': percentile('ttft_seconds', 50), 'p95_ttft_seconds': percentile('ttft_seconds', 95),
               'pim_route_fraction': (sum(row['selected_route'] == 'pim' for row in valid) / len(valid) if valid else None),
               'official_llm_judge': 'not_run; requires the configured MemBase judge endpoint',
               'f1_protocol': 'verified pinned MemBase TokenF1 functions executed unchanged',
               'task_scope': 'temporal MobileMem text QA; not interactive mobile task completion',
               'boundary': 'Chinese token-F1 is diagnostic only, never thresholded into task success; source-event hits do not establish span-level support; TTFT is reader-relative; failures stay in quality denominator'}
    output = {'summary': summary, 'rows': rows,
              'provenance': {'prepared_audit': json.loads((args.prepared / 'audit.json').read_text()),
                             'result_provenance': payload.get('provenance'), 'upstream': upstream,
                             'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest()},
              'judge_requests': judge_requests,
              'boundary': summary['boundary']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
