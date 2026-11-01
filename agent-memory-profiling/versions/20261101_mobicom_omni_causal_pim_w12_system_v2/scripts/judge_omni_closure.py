"""Evaluation-only Omni protocol using the explicitly selected Qwen3-VL-8B proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from audit_proof_direction import selected_jsonl


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def parse_label(response):
    try:
        parsed = json.loads(response)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or parsed.get('label') not in ('CORRECT', 'WRONG'):
        return None
    return parsed['label']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--prompt-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(
        '/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        raise ValueError('fresh judge output required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('labels.jsonl', 'manifests.jsonl'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared evaluation input hash mismatch')
    run_data = [(path, json.loads(path.read_text())) for path in args.runs]
    question_ids = {row['question_id'] for _, run in run_data for row in run.get('rows', [])}
    manifests = selected_jsonl(args.prepared / 'manifests.jsonl', question_ids)
    if any(row['split'] != 'development' for row in manifests.values()):
        raise ValueError('development-only evaluation; validation and holdout remain sealed')
    labels = selected_jsonl(args.prepared / 'labels.jsonl', question_ids)
    marker = 'Prompt for LLM-as-a-Judge Evaluation'
    source = args.prompt_source.read_text()
    if source.count(marker) != 1:
        raise ValueError('unexpected Omni prompt schema')
    template = source.split(marker, 1)[1].strip()
    requests = []
    run_hashes = {}
    for run_path, run in run_data:
        if not run.get('complete') or not run.get('rows'):
            raise ValueError('complete serving runs required; no survivor-only evaluation')
        run_hashes[str(run_path)] = digest(run_path)
        seen = set()
        for row in run['rows']:
            question_id = row['question_id']
            if question_id in seen:
                raise ValueError('duplicate question within serving run')
            seen.add(question_id)
            manifest = manifests[question_id]
            label = labels[question_id]
            if manifest['split'] != 'development':
                raise ValueError('this pilot evaluator is development-only; holdout remains sealed')
            if (hashlib.sha256(row['answer'].encode()).hexdigest() != row['answer_sha256']
                    or hashlib.sha256(manifest['question'].encode()).hexdigest() != row['question_sha256']):
                raise ValueError('question/prediction digest mismatch')
            replacements = {'[Question]': manifest['question'],
                            '[Gold Answer]': json.dumps(label['golden_answers'], ensure_ascii=False),
                            '[Evidences]': json.dumps(label['raw_evidence'], ensure_ascii=False),
                            '[Generated Answer]': row['answer']}
            sections = template.rsplit('QUESTION TO EVALUATE', 1)
            if len(sections) != 2:
                raise ValueError('unexpected Omni evaluation prompt structure')
            prompt = sections[0] + 'QUESTION TO EVALUATE\n' + '\n'.join(
                field + ': ' + replacements[placeholder] for field, placeholder in (
                    ('Question', '[Question]'), ('Gold answer', '[Gold Answer]'),
                    ('Evidence', '[Evidences]'), ('Generated answer', '[Generated Answer]')))
            prompt += '\nReturn only one JSON object with keys "reason" and "label". The label must be CORRECT or WRONG.'
            requests.append({'run': str(run_path), 'question_id': question_id,
                             'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                             'prediction_sha256': row['answer_sha256'],
                             'stop_reason': row['stop_reason']})
    provenance = {'judge_model': str(args.model), 'model_files_sha256': {
        path.name: digest(path) for path in sorted(args.model.iterdir())
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt')},
        'run_sha256': run_hashes, 'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'omni_prompt_source_sha256': digest(args.prompt_source), 'script_sha256': digest(Path(__file__)),
        'temperature': 0.0, 'max_new_tokens': args.max_new_tokens,
        'protocol': 'Omni strict coverage/evidence rubric + JSON-only output clarification',
        'boundary': 'user-selected Qwen3-VL-8B text-answer proxy; not official Qwen3-14B score, not visual grounding or agent execution success'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    registration = args.output.with_suffix('.registration.json')
    with registration.open('x') as handle:
        json.dump({'provenance': provenance, 'requests': requests}, handle, ensure_ascii=False, indent=2)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'rows': [], 'provenance': provenance}, handle, ensure_ascii=False)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(2)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to(args.device).eval()
    rows = []
    summaries = {}
    for request in requests:
        started = time.perf_counter()
        messages = [{'role': 'system', 'content': 'You evaluate answer correctness. Follow the supplied evidence rubric.'},
                    {'role': 'user', 'content': request['prompt']}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        inputs = processor(text=[rendered], return_tensors='pt').to(args.device)
        if inputs.input_ids.shape[1] > 32768:
            raise ValueError('judge input exceeds registered context; no silent evidence truncation')
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        response = processor.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        label = parse_label(response)
        rows.append({key: value for key, value in request.items() if key != 'prompt'} | {
            'judge_label': label, 'parse_valid': label is not None, 'response': response,
            'judge_seconds': time.perf_counter() - started})
        run_rows = [row for row in rows if row['run'] == request['run']]
        correct = sum(row['judge_label'] == 'CORRECT' for row in run_rows)
        valid = sum(row['parse_valid'] for row in run_rows)
        summaries[request['run']] = {'questions': len(run_rows), 'correct': correct,
            'invalid': len(run_rows) - valid, 'full_denominator_accuracy': correct / len(run_rows),
            'valid_judge_accuracy': correct / valid if valid else None,
            'official_judge_accuracy': None}
        args.output.write_text(json.dumps({'complete': False, 'rows': rows, 'summaries': summaries,
                                          'provenance': provenance}, ensure_ascii=False, indent=2))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps({'complete': True, 'rows': rows, 'summaries': summaries,
                                      'provenance': provenance}, ensure_ascii=False, indent=2))
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
