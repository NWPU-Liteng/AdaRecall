"""Charge known memory-index/state reservations before placing decode weights."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_omni_smoke_closure import digest
from pim_backend.decode_gemv import project_trace
from pim_backend.lpddr_bank_pim import BankPimConfig


def joint_projection(smoke, config, *, reader_tensor_bytes, embedding_dimension, device_capacity_bytes):
    if min(reader_tensor_bytes, embedding_dimension, device_capacity_bytes) <= 0:
        raise ValueError('positive Reader tensor size, embedding dimension and device capacity required')
    store = smoke['embedding_state_reuse']
    if not 0 <= store['peak_resident_bytes'] <= store['capacity_bytes']:
        raise ValueError('invalid observed embedding-store residency')
    dense_index_bytes = smoke['embedding_windows'] * embedding_dimension * 4
    visual_store = smoke.get('visual_state_reuse') or {}
    visual_budget = visual_store.get('capacity_bytes', 0)
    visual_peak = visual_store.get('peak_resident_bytes', 0)
    if visual_budget < 0 or visual_peak < 0 or visual_peak > visual_budget:
        raise ValueError('invalid visual state reservation')
    reserved_bytes = store['capacity_bytes'] + dense_index_bytes + visual_budget
    observed_lower_bound = reader_tensor_bytes + store['peak_resident_bytes'] + dense_index_bytes + visual_peak
    if reserved_bytes >= config.pim_capacity_bytes:
        raise ValueError('known index/state reservations exhaust PIM pool')
    weight_config = replace(config, pim_capacity_bytes=config.pim_capacity_bytes - reserved_bytes)
    projected = project_trace(smoke['operator_trace']['operators'], weight_config)
    known_pool_bytes = projected['resident_bytes'] + reserved_bytes
    return {'schema_version': 'omni-joint-capacity-lower-bound-v2',
            'device_capacity_bytes_assumed': device_capacity_bytes,
            'reader_checkpoint_tensor_bytes': reader_tensor_bytes,
            'known_reservations': {'dense_fp32_index_bytes': dense_index_bytes,
                                  'declared_embedding_state_budget_bytes': store['capacity_bytes'],
                                  'declared_visual_state_budget_bytes': visual_budget},
            'observed_embedding_state_peak_bytes': store['peak_resident_bytes'],
            'observed_visual_state_peak_bytes': visual_peak,
            'reservation_policy': 'declare full bounded embedding and optional visual stores before placement; no future observed occupancy oracle',
            'known_reservations_total_bytes': reserved_bytes,
            'pim_pool_capacity_bytes': config.pim_capacity_bytes,
            'weight_placement_budget_bytes': weight_config.pim_capacity_bytes,
            'placed_weight_bytes': projected['resident_bytes'],
            'known_joint_pool_bytes': known_pool_bytes,
            'known_joint_pool_fits': known_pool_bytes <= config.pim_capacity_bytes,
            'minimum_unique_device_bytes': observed_lower_bound,
            'declared_reader_index_state_bytes': reader_tensor_bytes + reserved_bytes,
            'reader_alone_fits_device': reader_tensor_bytes <= device_capacity_bytes,
            'minimum_unique_device_bytes_fit': observed_lower_bound <= device_capacity_bytes,
            'full_device_residency_qualified': False,
            'full_system_pim_feasibility_qualified': False,
            'end_to_end_speedup': None, 'decode_linear_projection': projected,
            'unaccounted_capacity': ['embedder weights', 'Reader KV/recurrent state',
                'vision activations', 'runtime temporary allocations', 'Python/index metadata',
                'memory-controller reservations', 'potential duplicate host/PIM weight layouts'],
            'boundary': 'known tensor reservations only; one fixed shared-pool scenario, not capacity tuning; incomplete lower bound cannot qualify device capacity; projected cycles are not measured server or mobile latency'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json')
    parser.add_argument('--device-capacity-bytes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh shared-capacity artifact required')
    smoke = json.loads(args.smoke.read_text())
    if not smoke.get('complete'):
        raise ValueError('complete Reader trace required')
    profile = json.loads(args.profile.read_text())
    reader_index_path = Path(smoke['reader']) / 'model.safetensors.index.json'
    reader_index = json.loads(reader_index_path.read_text())
    embedding_config_path = Path(smoke['embedder']) / 'config.json'
    if digest(embedding_config_path) != smoke['embedder_config_sha256']:
        raise ValueError('embedding configuration changed')
    dimension = json.loads(embedding_config_path.read_text())['hidden_size']
    result = joint_projection(smoke, BankPimConfig(**profile['config']),
                              reader_tensor_bytes=reader_index['metadata']['total_size'],
                              embedding_dimension=dimension,
                              device_capacity_bytes=args.device_capacity_bytes)
    result.update(smoke_sha256=digest(args.smoke), profile_sha256=digest(args.profile),
                  profile_calibration_status=profile['calibration_status'],
                  reader_index_sha256=digest(reader_index_path),
                  source_files_sha256={name: digest(ROOT / name) for name in (
                      'scripts/project_omni_joint_capacity.py', 'scripts/audit_omni_smoke_closure.py',
                      'pim_backend/decode_gemv.py', 'pim_backend/lpddr_bank_pim.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('known_reservations_total_bytes',
          'placed_weight_bytes', 'known_joint_pool_fits', 'reader_alone_fits_device',
          'full_system_pim_feasibility_qualified', 'end_to_end_speedup')}))


if __name__ == '__main__':
    main()
