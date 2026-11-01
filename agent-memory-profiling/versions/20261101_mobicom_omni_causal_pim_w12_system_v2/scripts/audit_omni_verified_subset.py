"""Pair an independently verified prefix subset with a complete timing stream."""
import argparse
import json
from pathlib import Path

from audit_omni_smoke_closure import audit_performance_pair, digest, verify_source_snapshot


def audit_subset(verified, performance):
    if not verified.get('complete') or not performance.get('complete'):
        raise ValueError('completed verified and performance executions required')
    verified_ids = [row['question_id'] for row in verified['rows']]
    if not verified_ids or len(set(verified_ids)) != len(verified_ids):
        raise ValueError('nonempty unique verification identities required')
    performance_ids = [row['question_id'] for row in performance['rows']]
    if len(set(performance_ids)) != len(performance_ids):
        raise ValueError('duplicate performance identity')
    if performance_ids[:len(verified_ids)] != verified_ids:
        raise ValueError('verification must replay the same cold-stream prefix, not selected warm requests')
    pair = audit_performance_pair(verified, {**performance, 'rows': performance['rows'][:len(verified_ids)]})
    probes = verified['operator_trace']['functional_samples']
    if len(probes) != 3 or not all(probe['allclose'] for probe in probes.values()):
        raise ValueError('actual-activation functional probe coverage failed')
    return {'schema_version': 'omni-verified-prefix-subset-audit-v1',
            'input_output_pair_integrity': pair['input_output_pair_integrity'],
            'verified_question_ids': verified_ids, 'verified_requests': len(verified_ids),
            'total_performance_requests': len(performance_ids),
            'full_performance_stream_generation_equivalence_qualified': len(verified_ids) == len(performance_ids),
            'functional_samples': probes,
            'boundary': 'only named cold-prefix requests were independently checked; verified probe timings excluded from performance; no all-cohort generation-equivalence or PIM acceleration claim'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('verified', 'performance', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh verified-subset audit required')
    verified, performance = json.loads(args.verified.read_text()), json.loads(args.performance.read_text())
    for path, record in ((args.verified, verified), (args.performance, performance)):
        registration = json.loads(path.with_suffix('.registration.json').read_text())
        if registration['source_files_sha256'] != record['source_files_sha256']:
            raise ValueError('runtime source differs from registration')
        if registration['question_ids'] != [row['question_id'] for row in record['rows']]:
            raise ValueError('runtime questions differ from registration')
        if not verify_source_snapshot(record, registration):
            raise ValueError('frozen source snapshot required')
    if verified['source_files_sha256'] != performance['source_files_sha256']:
        raise ValueError('verified and performance serving sources differ')
    result = audit_subset(verified, performance)
    result.update(verified_sha256=digest(args.verified), performance_sha256=digest(args.performance),
                  source_snapshot_verified=True,
                  source_files_sha256={name: digest(Path(__file__).parent / name) for name in (
                      'audit_omni_verified_subset.py', 'audit_omni_smoke_closure.py')})
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('input_output_pair_integrity',
          'verified_requests', 'total_performance_requests',
          'full_performance_stream_generation_equivalence_qualified')}))


if __name__ == '__main__':
    main()
