"""Register exact candidate code and model identities before generation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from omni_proof_runtime import file_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, required=True)
    parser.add_argument('--reranker', type=Path, required=True)
    parser.add_argument('--candidate', default='proof_recovery_contract_v2')
    args = parser.parse_args()
    runtime = args.root / 'runtime_mainline'
    models = {}
    for name in ('reader', 'embedder', 'reranker'):
        model = getattr(args, name)
        models[name] = {'path': str(model), 'files_sha256': {path.name: file_digest(path)
            for path in sorted(model.iterdir()) if path.is_file()
            and path.suffix in {'.json', '.safetensors', '.jinja', '.txt'}}}
    protocol = {'registered_before_generation': True, 'candidate': args.candidate,
        'prepared_audit_sha256': file_digest(args.prepared / 'audit.json'),
        'cohort_sha256': file_digest(args.root / 'cohort.json'),
        'source_files_sha256': {str(path.relative_to(runtime)): file_digest(path)
            for path in sorted(runtime.rglob('*')) if path.is_file() and path.suffix in {'.py', '.sh', '.json'}},
        'models': models, 'input_tokens': 8192, 'text_tokens': 3600, 'max_images': 4,
        'runtime_dependencies': {name: importlib.metadata.version(name)
                                 for name in ('torch', 'transformers', 'xgrammar')},
        'image_max_pixels': 262144, 'answer_max_new_tokens': 512,
        'parser_max_new_tokens': 1024, 'witness_max_new_tokens': 4096,
        'retrieval_per_demand': 48, 'planner_time_limit_seconds': 2,
        'source_contract_repair': 'query-only logical obligations; compact source indices; only source-located declared relation keys are required; unrelated optional fields cannot invalidate an anchored witness',
        'evidence_query_and_or_contract': 'query-only demand decomposition; required demands conjunctive; alternative witness disjunction; source-exact bindings and relation constraints',
        'source_granularity': 'one primary source atom/span or one original-image region per candidate; repeated image hashes share one physical Reader object',
        'failed_proof_policy': 'never relabel a top-1 record as proof; use a separately tagged nonproof budget-feasible retrieval package or empty evidence',
        'structured_decoding': 'XGrammar JSON-schema constrained autoregressive decoding; parser max 4 demands; witness max 8 records per witness',
        'sharding': [{'gpu': gpu, 'user': str(gpu % 3), 'shard_index': gpu // 3,
                      'shard_count': 3 if gpu % 3 == 0 else 2} for gpu in range(7)],
        'judge': 'unchanged Qwen3-VL-8B-Instruct strict JSON protocol',
        'evaluation_label_files_read': [], 'validation_accessed': False, 'holdout_accessed': False,
        'baseline_or_ablation_launched': False, 'main_experiment_closed': False,
        'measurement': 'shared-host server software screening; uncalibrated BF16 PIM disabled'}
    with (args.root / 'protocol.json').open('x') as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
