"""Create isolated label-side judge inputs for one frozen Omni smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from score_mobilemem import load_pinned_protocol


def build_requests(smoke, labels, manifests, template,
                   allowed_splits=('development',)):
    allowed_splits = set(allowed_splits)
    if not smoke.get('complete') or not smoke.get('rows'):
        raise ValueError('complete nonempty smoke required')
    requests, seen = [], set()
    for row in smoke['rows']:
        question_id = row['question_id']
        if question_id in seen:
            raise ValueError('duplicate prediction')
        seen.add(question_id)
        label, manifest = labels[question_id], manifests[question_id]
        if manifest['split'] not in allowed_splits:
            raise ValueError('split is outside the authorized evaluation scope')
        if (hashlib.sha256(row['answer'].encode()).hexdigest() != row['answer_sha256']
                or hashlib.sha256(manifest['question'].encode()).hexdigest() != row['question_sha256']):
            raise ValueError('prediction/question integrity failure')
        golden = label['golden_answers']
        if not golden:
            raise ValueError('missing reference answers')
        formatted_golden = golden[0] if len(golden) == 1 else f"[{', '.join(golden)}]"
        prompt = template.substitute(question=manifest['question'], prediction=row['answer'],
                                     golden_answers=formatted_golden)
        requests.append({'question_id': question_id, 'runtime_status': 'ok',
                         'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                         'prediction_sha256': row['answer_sha256'],
                         'reader_stop_reason': row['stop_reason']})
    return requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allowed-splits', nargs='+',
                        choices=('development', 'validation', 'holdout'),
                        default=['development'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh judge input artifact required')
    _, _, template, protocol = load_pinned_protocol()
    labels = {row['question_id']: row for row in (
        json.loads(line) for line in (args.prepared / 'labels.jsonl').read_text().splitlines())}
    manifests = {row['question_id']: row for row in (
        json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    smoke = json.loads(args.smoke.read_text())
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if hashlib.sha256((args.prepared / 'audit.json').read_bytes()).hexdigest() != smoke['prepared_audit_sha256']:
        raise ValueError('prepared audit changed')
    for name in ('labels.jsonl', 'manifests.jsonl'):
        if hashlib.sha256((args.prepared / name).read_bytes()).hexdigest() != audit['artifacts_sha256'][name]:
            raise ValueError('prepared inputs changed')
    requests = build_requests(smoke, labels, manifests, template,
                              allowed_splits=tuple(args.allowed_splits))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'judge_requests': requests,
        'smoke_sha256': hashlib.sha256(args.smoke.read_bytes()).hexdigest(),
        'prepared_audit_sha256': smoke['prepared_audit_sha256'],
        'allowed_splits': args.allowed_splits,
        'boundary': 'text-only answer correctness; no visual grounding evaluation'},
        ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'requests': len(requests), 'smoke_sha256': hashlib.sha256(args.smoke.read_bytes()).hexdigest(),
                      'evaluator_commit': protocol['commit'],
                      'boundary': 'evaluation-only input; labels never enter serving'}))


if __name__ == '__main__':
    raise SystemExit(main())
