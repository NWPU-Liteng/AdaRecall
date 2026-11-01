"""Isolated label-side quality audit after complete serving pair qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from build_omni_smoke_judge_requests import build_requests
from run_omni_registered_cohort import digest
from score_mobilemem import load_pinned_protocol


def summarize_quality(predictions, requests, judge):
    if not judge.get('complete') or not predictions or len(predictions) != len(requests):
        raise ValueError('complete whole-cohort judgments and nonempty predictions required')
    identities = [row['question_id'] for row in predictions]
    if (len(set(identities)) != len(identities)
            or [row['question_id'] for row in requests] != identities
            or [row['question_id'] for row in judge['rows']] != identities):
        raise ValueError('prediction/request/judgment whole-cohort order differs')
    rows = []
    for prediction, request, judgment in zip(predictions, requests, judge['rows']):
        if (prediction['answer_sha256'] != request['prediction_sha256']
                or prediction['answer_sha256'] != judgment['prediction_sha256']
                or request['prompt_sha256'] != judgment['prompt_sha256']
                or request['reader_stop_reason'] != prediction['stop_reason']
                or (judgment['parse_valid'] and judgment['value'] not in (0., 1.))):
            raise ValueError('judged answer/prompt identity or score contract differs')
        success = (judgment['parse_valid'] and judgment['value'] == 1.
                   and prediction['stop_reason'] == 'eos')
        rows.append({'question_id': prediction['question_id'], 'person_id': prediction['person_id'],
            'parse_valid': judgment['parse_valid'], 'local_judge_value': judgment['value'],
            'stop_reason': prediction['stop_reason'], 'strict_local_proxy_success': bool(success)})
    return {'requests': len(rows), 'rows': rows,
        'parse_valid_requests': sum(row['parse_valid'] for row in rows),
        'eos_requests': sum(row['stop_reason'] == 'eos' for row in rows),
        'strict_local_proxy_successes': sum(row['strict_local_proxy_success'] for row in rows),
        'strict_local_proxy_success_rate': sum(row['strict_local_proxy_success'] for row in rows) / len(rows),
        'boundary': 'independent local same-family 8B text-answer judge; development memory-QA proxy, not official accuracy, interactive mobile-agent task success or visual grounding; malformed judgments remain in denominator'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-audit', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    pair = json.loads(args.pair_audit.read_text())
    if pair['execution_integrity'] != 'pass' or args.output_dir.exists():
        raise ValueError('qualified complete serving pair and fresh evaluation directory required')
    plan_path = args.run_root / 'cohort_plan.json'
    plan = json.loads(plan_path.read_text())
    labels_path, manifests_path = args.prepared / 'labels.jsonl', args.prepared / 'manifests.jsonl'
    audit_path = args.prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if digest(audit_path) != plan['prepared_audit_sha256']:
        raise ValueError('prepared label/serving provenance differs')
    for path in (labels_path, manifests_path):
        if digest(path) != audit['artifacts_sha256'][path.name]:
            raise ValueError('evaluation input changed')
    labels = {row['question_id']: row for row in map(json.loads, labels_path.read_text().splitlines())}
    manifests = {row['question_id']: row for row in map(json.loads, manifests_path.read_text().splitlines())}
    _, _, template, protocol = load_pinned_protocol()
    predictions, requests = [], []
    provenance = {str(path): digest(path) for path in (args.pair_audit, plan_path, audit_path,
                                                     labels_path, manifests_path)}
    for job in plan['jobs']:
        path = Path(job['output'])
        if digest(path) != pair['inputs_sha256'][str(path)]:
            raise ValueError('qualified serving predictions changed before evaluation')
        run = json.loads(path.read_text())
        predictions.extend(run['rows'])
        requests.extend(build_requests(run, labels, manifests, template))
        provenance[str(path)] = digest(path)
    if len(predictions) != pair['requests']:
        raise ValueError('whole qualified pair denominator required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    requests_path = args.output_dir / 'judge_requests.json'
    with requests_path.open('x') as handle:
        json.dump({'judge_requests': requests, 'inputs_sha256': provenance,
            'evaluator_commit': protocol['commit'], 'evaluation_only': True}, handle, ensure_ascii=False, indent=2)
    judge_path = args.output_dir / 'judge.json'
    subprocess.run([sys.executable, str(Path(__file__).with_name('judge_mobilemem_local.py')),
                    '--scores', str(requests_path), '--output', str(judge_path)], check=True)
    judge = json.loads(judge_path.read_text())
    result = {'schema_version': 'omni-service-pair-quality-v1', 'complete': True,
        **summarize_quality(predictions, requests, judge),
        'inputs_sha256': {**provenance, str(requests_path): digest(requests_path),
                         str(judge_path): digest(judge_path)},
        'source_files_sha256': {str(path): digest(path) for path in (
            Path(__file__), Path(__file__).with_name('judge_mobilemem_local.py'),
            Path(__file__).with_name('build_omni_smoke_judge_requests.py'))},
        'serving_evaluation_label_files_read': [], 'evaluator_label_files_read': [str(labels_path)],
        'main_experiment_closed': False, 'publication_ready': False}
    with (args.output_dir / 'summary.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'eos_requests',
        'strict_local_proxy_successes', 'strict_local_proxy_success_rate')}))


if __name__ == '__main__':
    main()
