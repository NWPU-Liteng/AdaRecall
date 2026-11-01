#!/usr/bin/env python3
"""Project uniform 8/10/12-bit bit-sliced decode cost on the frozen workload."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import simulate_hyperpim_sc_timeline as timeline
from pim_backend.hyperpim_timeline import (
    HyperPimTimelineConfig, simulate_token, with_residual_bits, with_weight_bits)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--matrix-costs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--safety-factor', type=float, default=1.15)
    parser.add_argument('--dispatch-ms-per-token', type=float, default=4.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    if (not math.isfinite(args.safety_factor) or args.safety_factor < 1
            or not math.isfinite(args.dispatch_ms_per_token)
            or args.dispatch_ms_per_token < 0):
        raise ValueError('valid safety factor and dispatch cost required')

    projection = json.loads(args.projection.read_text())
    matrix_costs = json.loads(args.matrix_costs.read_text())
    sampled = timeline.decode_sequence(args.trace_root)
    sequence = sampled['sequence']
    weight_elements = sum(item['inputs'] * item['outputs'] for item in sequence)
    if weight_elements != 4_204_789_760:
        raise ValueError('decoder weight inventory differs from the frozen trace')

    base = HyperPimTimelineConfig()
    calibration = timeline.calibrate(sequence, matrix_costs, base)
    if not calibration['passed']:
        raise ValueError('INT8 cycle model failed the GeniePIM calibration gate')
    nonlinear = float(projection['gpu_nonlinear_profile']['median_ms_per_token'])
    grammar = float(projection['pim_grammar_model']['milliseconds_per_token'])
    dispatch = args.dispatch_ms_per_token

    arms = {}
    for name, config, arm_dispatch in (
            ('w8a16_r0', with_residual_bits(base, 0), dispatch),
            ('w8a16_r2', with_residual_bits(base, 2), dispatch),
            ('w10a16', with_weight_bits(base, 10), dispatch),
            ('w12a16', with_weight_bits(base, 12), dispatch),
            ('w12a16_joint_worst', replace(
                with_weight_bits(base, 12), conversion_lanes_per_bank=8,
                bf16_reconstruction_lanes_per_bank=8,
                bf16_round_lanes_per_bank=8,
                bf16_group_reduction_lanes_per_bank=8,
                bf16_macs_per_bank_cycle=1,
                refresh_block_cycles=2 * base.refresh_block_cycles,
                t_rcd_cycles=2 * base.t_rcd_cycles,
                t_rp_cycles=2 * base.t_rp_cycles,
                layer_fence_cycles=2 * base.layer_fence_cycles), 10.0)):
        simulated = simulate_token(sequence, config)
        pim_ms = simulated['totals']['pim_linear_ms_per_token']
        assessment = timeline.assess_target(
            projection, pim_ms, nonlinear, grammar, arm_dispatch,
            args.safety_factor)
        arms[name] = {
            'weight_bits': config.weight_bits,
            'residual_bits': config.residual_bits,
            'bf16_reconstruct_weights': config.bf16_reconstruct_weights,
            'pipeline_bf16_reconstruction_and_mac': (
                config.pipeline_bf16_reconstruction_and_mac),
            'bf16_macs_per_bank_cycle': config.bf16_macs_per_bank_cycle,
            'bf16_reconstruction_lanes_per_bank': (
                config.bf16_reconstruction_lanes_per_bank),
            'bf16_round_lanes_per_bank': config.bf16_round_lanes_per_bank,
            'bf16_group_reduction_lanes_per_bank': (
                config.bf16_group_reduction_lanes_per_bank),
            'dispatch_ms_per_token': arm_dispatch,
            'pim_linear_ms_per_token': pim_ms,
            'structured_ms_per_token': assessment['structured_ms_per_token'],
            'reader_ms_per_token': assessment['reader_ms_per_token'],
            'speedup': assessment['speedup'],
            'target': assessment['target'],
            'control': assessment['control'],
            'totals': simulated['totals'],
        }

    result = {
        'schema_version': 'w12-bank-timeline-projection-v2',
        'complete': True,
        'requests': len(projection['requests_detail']),
        'boundary': (
            'cycle-level target-architecture projection on a measured server trace; '
            'W12A16 includes explicit 12-bit reconstruction, BF16 rounding, BF16 MAC '
            'and group reduction cycles; not physical PIM hardware, phone measurement, '
            'or a quality-qualified E2E result'),
        'inputs_sha256': {
            str(args.projection): digest(args.projection),
            str(args.matrix_costs): digest(args.matrix_costs),
            sampled['path']: digest(sampled['path']),
        },
        'decode_sequence': {
            'source': sampled['path'],
            'question_id': sampled['question_id'],
            'steps': sampled['steps'],
            'calls_per_token': len(sequence),
            'weight_elements': weight_elements,
        },
        'calibration': calibration,
        'arms': arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({
        name: {
            'pim_linear_ms_per_token': arm['pim_linear_ms_per_token'],
            'service_vs_host_control': arm['speedup'][
                'service_vs_host_control_lower_bound'],
            'ttft_vs_host_control': arm['speedup'][
                'ttft_vs_host_control_lower_bound'],
            'service_p95_vs_host_control': arm['speedup'][
                'service_p95_vs_host_control_lower_bound'],
            'ttft_p95_vs_host_control': arm['speedup'][
                'ttft_p95_vs_host_control_lower_bound'],
        }
        for name, arm in arms.items()
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
