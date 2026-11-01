"""Summarize a frozen development pilot using the fixed local Judge outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row['question_id']
            if identity in rows:
                raise ValueError(f'duplicate question ID in {path}: {identity}')
            rows[identity] = row
    return rows


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999)))
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    correct = sum(row['judge_label'] == 'CORRECT' for row in rows)
    valid = sum(row['parse_valid'] for row in rows)
    return {
        'questions': count,
        'judge_correct': correct,
        'judge_invalid': count - valid,
        'success_rate_all_questions': correct / count if count else None,
        'accuracy_valid_judges_only': correct / valid if valid else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('labels.jsonl', 'manifests.jsonl'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError(f'prepared evaluation input hash mismatch: {name}')
    cohort = json.loads(args.cohort.read_text())
    if cohort.get('schema') != 'mobilemem-omni-frozen-pilot-cohort-v1':
        raise ValueError('frozen pilot cohort required')
    expected = {identity for values in cohort['question_ids_by_user'].values()
                for identity in values}
    if len(expected) != cohort['questions']:
        raise ValueError('pilot cohort question count is inconsistent')

    predictions = {}
    run_paths = []
    for user in cohort['question_ids_by_user']:
        path = args.run_root / f'user_{user}_shard_0.json'
        run = json.loads(path.read_text())
        if not run.get('complete'):
            raise ValueError(f'incomplete Reader output: {path}')
        run_paths.append(path)
        for row in run.get('rows', []):
            identity = row['question_id']
            if identity in predictions:
                raise ValueError(f'duplicate Reader output: {identity}')
            predictions[identity] = row
    if set(predictions) != expected:
        raise ValueError('Reader question coverage differs from the frozen pilot cohort')

    judge = json.loads(args.judge.read_text())
    if not judge.get('complete'):
        raise ValueError('complete Judge output required')
    judged = {}
    for row in judge.get('rows', []):
        identity = row['question_id']
        if identity in judged:
            raise ValueError(f'duplicate Judge output: {identity}')
        judged[identity] = row
    if set(judged) != expected:
        raise ValueError('Judge question coverage differs from the frozen pilot cohort')

    labels = read_jsonl(args.prepared / 'labels.jsonl')
    manifests = read_jsonl(args.prepared / 'manifests.jsonl')
    if not expected <= set(labels) or not expected <= set(manifests):
        raise ValueError('pilot evaluation labels or manifests are incomplete')
    if any(manifests[identity]['split'] != 'development' for identity in expected):
        raise ValueError('pilot includes a non-development question')

    rows = []
    for identity in sorted(expected):
        prediction, score, label = predictions[identity], judged[identity], labels[identity]
        rows.append({
            'question_id': identity,
            'person_id': label['person_id'],
            'question_type': label['question_type'],
            'image_evidence_linked': bool(label.get('evidence_image_refs')),
            'judge_label': score.get('judge_label'),
            'parse_valid': bool(score.get('parse_valid')),
            'service_seconds': float(prediction['service_seconds']),
            'judge_seconds': float(score.get('judge_seconds', 0.0)),
            'stop_reason': prediction.get('stop_reason'),
            'reader_answer_sha256': prediction['answer_sha256'],
        })
    if any(row['judge_label'] not in ('CORRECT', 'WRONG') and row['parse_valid']
           for row in rows):
        raise ValueError('Judge parse validity and labels disagree')

    categories = defaultdict(list)
    modalities = defaultdict(list)
    for row in rows:
        categories[row['question_type']].append(row)
        modalities['image_evidence_linked' if row['image_evidence_linked']
                   else 'no_image_evidence_link'].append(row)
    service = [row['service_seconds'] for row in rows]
    payload = {
        'schema': 'mobilemem-omni-main-pilot-evaluation-v1',
        'cohort_sha256': digest(args.cohort),
        'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'reader_runs_sha256': {str(path): digest(path) for path in run_paths},
        'judge_sha256': digest(args.judge),
        'judge_model': judge.get('provenance', {}).get('judge_model'),
        'judge_boundary': 'user-selected Qwen3-VL-8B-Instruct; local proxy, not official Qwen3-14B score',
        'registered_questions': len(expected),
        'reader_completed': len(predictions),
        'judge_completed': len(judged),
        'overall': summarize(rows),
        'by_question_type': {key: summarize(value) for key, value in sorted(categories.items())},
        'by_image_evidence_link': {key: summarize(value) for key, value in sorted(modalities.items())},
        'reader_service_latency_seconds': {
            'mean': sum(service) / len(service) if service else None,
            'p50': percentile(service, 0.50),
            'p95': percentile(service, 0.95),
        },
        'rows': rows,
        'boundary': '200 frozen development requests; labels/evidence read only after serving for Judge/evaluation; image-evidence linkage is not itself proof of visual grounding',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    report = [
        '# MobileMem-Omni 200 题 Pilot', '',
        f"- Reader 完成：{payload['reader_completed']}/{payload['registered_questions']}。",
        f"- Qwen3-VL-8B Judge 成功率：{payload['overall']['judge_correct']}/{payload['registered_questions']} = {payload['overall']['success_rate_all_questions']:.1%}（无效 Judge 按失败计入分母）。",
        f"- Judge 有效输出准确率：{payload['overall']['judge_correct']}/{payload['registered_questions'] - payload['overall']['judge_invalid']} = {payload['overall']['accuracy_valid_judges_only']:.1%}；无效输出 {payload['overall']['judge_invalid']}。",
        f"- Reader 服务时延：均值 {payload['reader_service_latency_seconds']['mean']:.1f}s，P50 {payload['reader_service_latency_seconds']['p50']:.1f}s，P95 {payload['reader_service_latency_seconds']['p95']:.1f}s。",
        '', '## 多模态子集',
    ]
    for name, values in payload['by_image_evidence_link'].items():
        report.append(f"- `{name}`：{values['judge_correct']}/{values['questions']}，成功率 {values['success_rate_all_questions']:.1%}。")
    report.extend(['', '## 七类任务'])
    for name, values in payload['by_question_type'].items():
        report.append(f"- `{name}`：{values['judge_correct']}/{values['questions']}，成功率 {values['success_rate_all_questions']:.1%}。")
    report.extend(['', '该结果是固定本地 Judge 的 200 题 development pilot，不是官方 Qwen3-14B 分数或最终 holdout 结论。'])
    (args.output.parent / 'report_200_pilot.md').write_text('\n'.join(report) + '\n')
    print(json.dumps({key: payload[key] for key in
                      ('overall', 'by_image_evidence_link', 'by_question_type',
                       'reader_service_latency_seconds')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
