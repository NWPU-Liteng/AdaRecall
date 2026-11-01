"""Short paired probe of constrained versus free structured decode control overhead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from omni_proof_query_contract import QUERY_CONTRACT_PROMPT, QUERY_CONTRACT_SCHEMA
from omni_proof_runtime import load_reader, structured_call
from residual_profile import ReaderOperatorTrace


def run_case(reader, processor, question, *, schema, max_new_tokens):
    trace = ReaderOperatorTrace(reader)
    try:
        result = structured_call(
            reader,
            processor,
            QUERY_CONTRACT_PROMPT,
            {'question': question, 'output_contract': 'JSON only'},
            max_tokens=max_new_tokens,
            budget=8192,
            json_schema=schema,
            trace=trace,
            profile_operator_kernels=False,
        )
    finally:
        trace.close()
    summary = trace.as_dict()
    return {
        'constrained': schema is not None,
        'stop_reason': result['stop_reason'],
        'input_tokens': result['structured_input_tokens'],
        'output_tokens': len(result['answer_token_ids']),
        'prefill_seconds': result['structured_reader_prefill_seconds'],
        'decode_seconds': result['structured_reader_decode_seconds'],
        'decode_phase_profile': result['decode_phase_profile'],
        'grammar_compile_seconds': result['structured_grammar_compile_seconds'],
        'linear_trace': {key: value for key, value in summary.items() if key != 'operators'},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 8:
        raise ValueError('at least eight generated tokens required')
    manifests = [json.loads(line) for line in
                 (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    manifest = next((row for row in manifests if row['question_id'] == args.question_id), None)
    if manifest is None or manifest.get('split') != 'development':
        raise ValueError('one authorized development question required')
    processor, reader = load_reader(args.reader)
    free = run_case(reader, processor, manifest['question'], schema=None,
                    max_new_tokens=args.max_new_tokens)
    constrained = run_case(reader, processor, manifest['question'], schema=QUERY_CONTRACT_SCHEMA,
                           max_new_tokens=args.max_new_tokens)
    free_ms = 1000 * free['decode_seconds'] / free['output_tokens']
    constrained_ms = 1000 * constrained['decode_seconds'] / constrained['output_tokens']
    result = {
        'schema': 'hyperpim-sc-structured-decode-phase-probe-v1',
        'question_id': args.question_id,
        'reader': str(args.reader),
        'max_new_tokens': args.max_new_tokens,
        'free': free,
        'constrained': constrained,
        'per_token_ms': {'free': free_ms, 'constrained': constrained_ms,
                         'constrained_over_free': constrained_ms / free_ms},
        'boundary': ('short synthetic truncation probe; both cases stop at the same token budget; '
                     'no answer-quality or service-latency claim'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                      ('question_id', 'per_token_ms', 'free', 'constrained')},
                     ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
