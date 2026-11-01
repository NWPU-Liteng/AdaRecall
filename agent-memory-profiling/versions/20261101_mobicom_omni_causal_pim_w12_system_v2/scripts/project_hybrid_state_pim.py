"""Project the unified hybrid-state PIM route from frozen diagnostics.

The inputs are label-free diagnostic artifacts only.  This script does not run
the Reader, read evaluation labels, or turn a modeled step ratio into E2E
speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.hybrid_state import HybridStateLayout, HybridStatePimPlanner
from pim_backend.lpddr_bank_pim import BankPimConfig


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _load_profile(path: Path) -> BankPimConfig:
    payload = json.loads(path.read_text())
    config = payload.get('config', payload)
    fields = BankPimConfig.__dataclass_fields__
    return BankPimConfig(**{name: config[name] for name in fields if name in config})


def project(native_state: Path, decode_components: Path, closure: Path,
            profile: Path, output: Path, active_agents: int,
            reserved_bytes: int, full_attention_query_heads: int) -> dict:
    native = json.loads(native_state.read_text())
    components = json.loads(decode_components.read_text())
    closure_payload = json.loads(closure.read_text())
    inventory = native.get('cache_inventory')
    if not isinstance(inventory, dict):
        raise ValueError('native state artifact must contain cache_inventory')
    layout = HybridStateLayout.from_cache_inventory(
        inventory, full_attention_query_heads=full_attention_query_heads)
    pim_config = _load_profile(profile)
    planner = HybridStatePimPlanner(pim_config, reserved_bytes=reserved_bytes)
    linear_plan = planner.plan(layout, active_agents=active_agents)
    plan = planner.unified_plan(layout, active_agents=active_agents)
    decode_derived = components.get('derived', {})
    closure_components = closure_payload.get('decode_components', {})
    result = {
        'schema': 'hybrid-state-pim-unified-qualification-v1',
        'inputs': {
            'native_state_sha256': digest(native_state),
            'decode_components_sha256': digest(decode_components),
            'closure_sha256': digest(closure),
            'pim_profile_sha256': digest(profile),
        },
        'native_state': {
            'target_tokens': native.get('target_tokens'),
            'total_bytes': inventory.get('total_bytes'),
            'full_attention_bytes': layout.full_attention_bytes,
            'full_attention_layers': layout.full_attention_layers,
            'full_attention_context_tokens': layout.full_attention_context_tokens,
            'full_attention_query_heads': layout.full_attention_query_heads,
            'full_attention_kv_heads': layout.full_attention_kv_heads,
            'full_attention_head_dim': layout.full_attention_head_dim,
            'linear_attention_state_bytes': layout.linear_state_bytes,
            'linear_layers': layout.linear_layers,
            'recurrent_heads': layout.recurrent_heads,
            'key_dim': layout.key_dim,
            'value_dim': layout.value_dim,
        },
        'pim_plan': plan,
        'linear_only_reference_plan': linear_plan,
        'pim_profile': {
            'clock_hz': pim_config.clock_hz,
            'bank_count': pim_config.bank_count,
            'capacity_bytes': pim_config.pim_capacity_bytes,
            'external_service_bytes_per_cycle': pim_config.external_service_bytes_per_cycle,
            'external_service_decimal_gb_s': (
                pim_config.external_service_bytes_per_cycle * pim_config.clock_hz / 1e9
            ),
            'calibration_status': json.loads(profile.read_text()).get(
                'calibration_status', 'unknown'),
        },
        'decode_materiality': {
            'measured_cuda_ms_mean': components.get(
                'steady_baseline_excluding_first_token', {}).get('cuda_ms', {}).get('mean'),
            'hybrid_state_upper_fraction_from_component_profile': decode_derived.get(
                'extended_pim_fraction_upper_mean'),
            'hybrid_state_upper_fraction_from_closure': closure_components.get(
                'hybrid_state_upper_mean'),
            'ideal_zero_cost_decode_upper_bound': (
                1.0 / (1.0 - float(decode_derived['extended_pim_fraction_upper_mean']))
                if decode_derived.get('extended_pim_fraction_upper_mean') is not None
                and float(decode_derived['extended_pim_fraction_upper_mean']) < 1.0 else None),
            'boundary': (
                'component upper bound only; it deletes eligible work and is not '
                'the unified PIM step or an E2E claim'),
        },
        'quality_and_execution': {
            'native_gpu_vs_host_exact_output_rate': closure_payload.get(
                'gate_values', {}).get('gpu_vs_host_exact_output_rate'),
            'physical_pim_executed': False,
            'evaluation_label_files_read': [],
            'current_200_question_path_changed': False,
        },
        'claim_boundary': (
            'label-free unified native-state PIM qualification; full-attention KV '
            'and linear recurrent/conv state share one LPDDR capacity tier, while '
            'model projections, softmax, cross-bank reduction, MLPs and semantic '
            'Reader work remain Host/GPU-owned. No measured or end-to-end speedup '
            'is claimed.'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-state', type=Path, required=True)
    parser.add_argument('--decode-components', type=Path, required=True)
    parser.add_argument('--closure', type=Path, required=True)
    parser.add_argument('--pim-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--active-agents', type=int, default=32)
    parser.add_argument('--reserved-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--full-attention-query-heads', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    result = project(args.native_state, args.decode_components, args.closure,
                     args.pim_profile, args.output, args.active_agents,
                     args.reserved_bytes, args.full_attention_query_heads)
    print(json.dumps({
        'output': str(args.output),
        'linear_attention_state_bytes': result['native_state']['linear_attention_state_bytes'],
        'full_hybrid_state_bytes': result['pim_plan']['admission']['per_agent_state_bytes'],
        'resident_agents': result['pim_plan']['admission']['resident_agents'],
        'admitted': result['pim_plan']['admission']['admitted'],
        'conditional_step_speedup_if_capacity_admitted': result['pim_plan'][
            'conditional_modeled_step_speedup_for_admitted_resident_state'],
        'per_state_step_ratio_diagnostic_not_workload_speedup': result['pim_plan'][
            'per_state_modeled_step_ratio_diagnostic_not_workload_speedup'],
        'capacity_scale_factor_required': result['pim_plan'][
            'capacity_scale_factor_required'],
        'ideal_zero_cost_decode_upper_bound': result['decode_materiality'][
            'ideal_zero_cost_decode_upper_bound'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
