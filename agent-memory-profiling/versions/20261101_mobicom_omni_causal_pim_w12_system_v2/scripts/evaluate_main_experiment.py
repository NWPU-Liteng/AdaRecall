"""Evaluate completed main-run predictions without touching serving code."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


TYPE_NAMES = {
    'single_hop': 'single_hop',
    'multi_hop': 'multi_hop',
    'knowledge_update': 'knowledge_update',
    'temporal_reasoning': 'temporal_reasoning',
    'visual_reasoning': 'visual_reasoning',
    'implicit_preference': 'implicit_preference',
    'abstention': 'abstention',
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(value: str) -> list[str]:
    value = re.sub(r'[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', ' ', str(value).lower())
    return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', value)


def f1(pred: str, reference: str) -> float:
    prediction, target = Counter(tokens(pred)), Counter(tokens(reference))
    if not prediction and not target:
        return 1.0
    if not prediction or not target:
        return 0.0
    overlap = sum((prediction & target).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(prediction.values())
    recall = overlap / sum(target.values())
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, references: list[str]) -> float:
    return max((f1(prediction, reference) for reference in references), default=0.0)


def load_predictions(root: Path) -> dict[str, dict]:
    predictions = {}
    for path in sorted(root.glob('user_*_shard_*.json')):
        payload = json.loads(path.read_text())
        if not payload.get('complete'):
            continue
        for row in payload.get('rows', []):
            identity = row['question_id']
            if identity in predictions and predictions[identity] != row:
                raise ValueError(f'duplicate prediction differs: {identity}')
            predictions[identity] = row
    return predictions


def load_labels(path: Path) -> dict[str, dict]:
    labels = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row['question_id'] in labels:
                    raise ValueError(f'duplicate label: {row["question_id"]}')
                labels[row['question_id']] = row
    return labels


def evaluate(prepared: Path, run_root: Path, output: Path, judge_model: Path | None) -> dict:
    labels = load_labels(prepared / 'labels.jsonl')
    predictions = load_predictions(run_root)
    rows, missing = [], []
    for question_id, label in labels.items():
        prediction = predictions.get(question_id)
        if prediction is None:
            missing.append(question_id)
            continue
        references = [str(value) for value in label.get('golden_answers', [''])]
        rows.append({
            'question_id': question_id,
            'person_id': label['person_id'],
            'question_type': label.get('question_type'),
            'prediction': prediction.get('answer', ''),
            'f1': best_f1(prediction.get('answer', ''), references),
            'prediction_sha256': prediction.get('answer_sha256'),
            'service_seconds': prediction.get('service_seconds'),
            'prefill_seconds': prediction.get('prefill_seconds'),
            'decode_seconds': prediction.get('decode_seconds'),
            'planning_seconds': prediction.get('proof_plan', {}).get('planning_seconds'),
            'evidence_objects': len(prediction.get('materialized_object_ids', [])),
            'images': len(prediction.get('images', [])),
        })
    overall = {
        'questions_labeled': len(labels),
        'questions_completed': len(rows),
        'questions_missing': len(missing),
        'coverage': len(rows) / len(labels) if labels else 0.0,
        'f1': sum(row['f1'] for row in rows) / len(rows) if rows else 0.0,
    }
    by_type = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[TYPE_NAMES.get(row['question_type'], str(row['question_type']))].append(row)
    for name in sorted(TYPE_NAMES.values()):
        values = grouped.get(name, [])
        by_type[name] = {'count': len(values),
                         'f1': sum(row['f1'] for row in values) / len(values) if values else 0.0}
    judge_status = 'not_requested'
    if judge_model is not None:
        judge_status = 'available_not_run' if judge_model.is_dir() else 'missing_qwen3_vl_8b_instruct'
    payload = {
        'schema': 'mobilemem-omni-main-evaluation-v1',
        'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'run_files': sorted(path.name for path in run_root.glob('user_*_shard_*.json')),
        'labels_sha256': digest(prepared / 'labels.jsonl'),
        'overall': overall,
        'by_question_type': by_type,
        'judge': {'status': judge_status,
                  'model': str(judge_model) if judge_model else None,
                  'boundary': 'fixed user-selected local Qwen3-VL-8B-Instruct Judge; not the official Qwen3-14B score'},
        'missing_question_ids': missing,
        'rows': rows,
        'closure': False,
        'boundary': 'labels are read only in evaluation; lexical F1 is not official Judge accuracy',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    report_lines = [
        '# MobileMem-Omni 主实验报告',
        '',
        '## 完成状态',
        f"- 标注问题总数：{overall['questions_labeled']}；已完成生成：{overall['questions_completed']}；缺失：{overall['questions_missing']}。",
        f"- Serving 覆盖率：{overall['coverage']:.4f}；词法 F1：{overall['f1']:.4f}。",
        f"- Qwen3-VL-8B-Instruct Judge：{judge_status}；当前 closure：{payload['closure']}。",
        '',
        '## 七类任务',
    ]
    for name in sorted(by_type):
        report_lines.append(f"- `{name}`：{by_type[name]['count']} 题，F1={by_type[name]['f1']:.4f}。")
    report_lines.extend([
        '',
        '## 解释边界',
        '- 超图通过 source-fact、变量绑定、来源见证和唯一物理对象物化参与证据恢复；不是仅保存 session 字段。',
        '- PIM 成本由同一请求轨迹另行建模；没有校准 LPDDR-PIM 硬件、完整 Reader 驻留和手机实测时，不报告真实设备或端到端 speedup。',
        '- labels、答案和证据只在本评测入口读取；它们没有进入 serving query、索引或证据选择。',
        '- Judge 缺失或问题未完成时，不把 lexical F1 写成官方回答得分。',
    ])
    (output.parent / 'report.md').write_text('\n'.join(report_lines) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judge-model', type=Path)
    args = parser.parse_args()
    payload = evaluate(args.prepared, args.run_root, args.output, args.judge_model)
    print(json.dumps({key: payload[key] for key in ('overall', 'by_question_type', 'judge', 'closure')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
