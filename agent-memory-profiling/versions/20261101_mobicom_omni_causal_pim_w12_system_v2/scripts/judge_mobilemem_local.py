"""Development-only semantic diagnosis using the pinned MemBase judge prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from score_mobilemem import load_pinned_protocol


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def reserve_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as handle:
            json.dump({'complete': False, 'rows': [], 'status': 'reserved_before_model_execution'}, handle)
    except FileExistsError as error:
        raise ValueError('fresh local diagnosis output required') from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    reserve_output(args.output)
    torch.set_num_threads(2)
    _, _, _, upstream = load_pinned_protocol()
    scores = json.loads(args.scores.read_text())
    requests = scores['judge_requests']
    if len({item['question_id'] for item in requests}) != len(requests):
        raise ValueError('duplicate judge input')
    model_hashes = {path.name: digest(path) for path in sorted(args.model.glob('*'))
                    if path.suffix in {'.safetensors', '.json'} and path.is_file()}
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to(args.device).eval()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in requests:
        started = time.perf_counter()
        if hashlib.sha256(item['prompt'].encode()).hexdigest() != item['prompt_sha256']:
            raise ValueError('judge prompt mutation')
        if item['runtime_status'] != 'ok':
            response = 'No'
        else:
            messages = [{'role': 'user', 'content': item['prompt']}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            inputs = processor(text=[text], return_tensors='pt').to(args.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            response = processor.tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        parsed = response.lower().strip().rstrip('.!。！')
        valid = parsed in {'yes', 'no'}
        rows.append({'question_id': item['question_id'], 'raw_response': response,
                     'parse_valid': valid, 'upstream_yes_substring_value': float('yes' in response.lower()),
                     'value': float('yes' in response.lower()) if valid else None,
                     'prompt_sha256': item['prompt_sha256'], 'prediction_sha256': item['prediction_sha256'],
                     'judge_seconds': time.perf_counter() - started})
        output = {'complete': False, 'rows': rows}
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    valid_rows = [row for row in rows if row['parse_valid']]
    output = {'complete': True, 'rows': rows,
              'summary': {'questions': len(rows), 'invalid_judgments': len(rows) - len(valid_rows),
                          'local_semantic_answer_accuracy': sum(row['value'] for row in valid_rows) / len(rows)
                          if rows and len(valid_rows) == len(rows) else None,
                          'official_reference_judge_accuracy': None},
              'provenance': {'scores_sha256': digest(args.scores), 'model': str(args.model),
                             'model_hashes': model_hashes, 'upstream_commit': upstream['commit'],
                             'script_sha256': digest(Path(__file__)), 'temperature': 0.0,
                             'max_new_tokens': 16, 'torch': torch.__version__, 'python': sys.version},
              'boundary': 'development diagnosis only; official default-exact-match prompt and yes parser, but Qwen3-VL-8B replaces reference judge; same-family bias and temporal protocol deviation; not mobile task success or publication-final accuracy'}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output['summary']), flush=True)
    return int(len(valid_rows) != len(rows))


if __name__ == '__main__':
    raise SystemExit(main())
