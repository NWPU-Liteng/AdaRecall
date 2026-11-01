"""Replay a frozen state-churn workload with a modeled hybrid-state PIM tier.

This is a trace replay, not a new Reader run.  The measured Host access records
are retained; only the eligible hybrid-state component is replaced by the
phase-level PIM model.  The output therefore cannot be used as measured E2E
latency or physical PIM speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError('nonempty values required')
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def replay(churn_path: Path, qualification_path: Path,
           decode_components_path: Path, output: Path) -> dict:
    churn = json.loads(churn_path.read_text())
    qualification = json.loads(qualification_path.read_text())
    decode_components = json.loads(decode_components_path.read_text())
    if churn.get('status') != 'complete' or not churn.get('access_records'):
        raise ValueError('complete state-churn access records required')
    plan = qualification.get('pim_plan', {})
    admission = plan.get('admission', {})
    if not admission.get('admitted'):
        raise ValueError('PIM state capacity admission failed for replay workload')
    fraction = decode_components.get('derived', {}).get('extended_pim_fraction_upper_mean')
    if fraction is None or not 0 <= float(fraction) < 1:
        raise ValueError('finite hybrid-state upper fraction required')
    pim_step = plan.get('pim_step', {})
    clock_hz = float(qualification.get('pim_profile', {}).get('clock_hz', 0.0))
    if not clock_hz:
        # The fixed LPDDR profile is registered in the qualification artifact's
        # source and the phase model uses its checked-in 800 MHz clock.
        clock_hz = 800_000_000.0
    pim_step_seconds = float(pim_step['cycles']) / clock_hz
    rows = []
    for access in churn['access_records']:
        baseline_total = float(access['total_seconds'])
        baseline_decode = float(access['decode_seconds'])
        residual_decode = baseline_decode * (1.0 - float(fraction))
        modeled_pim_decode = residual_decode + pim_step_seconds
        rows.append({
            'access_index': access['access_index'],
            'agent_index': access['agent_index'],
            'cycle_index': access['cycle_index'],
            'cache_hit': access['cache_hit'],
            'baseline_total_seconds': baseline_total,
            'baseline_decode_seconds': baseline_decode,
            'modeled_pim_decode_seconds': modeled_pim_decode,
            'modeled_pim_total_seconds': modeled_pim_decode,
            'removed_host_restore_seconds': float(access['restore_seconds']),
            'removed_host_eviction_seconds': float(access['eviction_seconds']),
        })
    baseline = [row['baseline_total_seconds'] for row in rows]
    candidate = [row['modeled_pim_total_seconds'] for row in rows]
    summary = {
        'accesses': len(rows),
        'baseline_total_seconds': sum(baseline),
        'modeled_pim_total_seconds': sum(candidate),
        'baseline_mean_seconds': statistics.fmean(baseline),
        'modeled_pim_mean_seconds': statistics.fmean(candidate),
        'baseline_p95_seconds': percentile(baseline, 0.95),
        'modeled_pim_p95_seconds': percentile(candidate, 0.95),
        'modeled_total_speedup': sum(baseline) / sum(candidate) if sum(candidate) else None,
        'modeled_mean_speedup': statistics.fmean(baseline) / statistics.fmean(candidate),
        'modeled_p95_speedup': percentile(baseline, 0.95) / percentile(candidate, 0.95),
    }
    result = {
        'schema': 'hybrid-state-pim-workload-replay-v1',
        'inputs': {
            'churn_sha256': digest(churn_path),
            'qualification_sha256': digest(qualification_path),
            'decode_components_sha256': digest(decode_components_path),
        },
        'workload': {
            'schedule': churn.get('schedule'),
            'agent_count': churn.get('agent_count'),
            'target_tokens': churn.get('target_tokens'),
            'hbm_capacity_agents': churn.get('hbm_capacity_agents'),
            'accesses': len(rows),
        },
        'model': {
            'hybrid_state_upper_fraction': float(fraction),
            'pim_step_cycles': pim_step.get('cycles'),
            'pim_step_seconds': pim_step_seconds,
            'pim_clock_hz': clock_hz,
            'physical_pim_executed': False,
            'measured_candidate_run': False,
        },
        'summary': summary,
        'rows': rows,
        'boundary': (
            'measured state-churn trace replay with one eligible hybrid-state phase '
            'substituted by an uncalibrated PIM model; not physical PIM, not a '
            'measured E2E pair, and not transferable to the current 200-question '
            'stateless mainline'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--churn', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--decode-components', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    result = replay(args.churn, args.qualification, args.decode_components, args.output)
    print(json.dumps({'output': str(args.output), **result['summary']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
