"""Project known Omni state residency against explicit device-memory budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from audit_omni_smoke_closure import digest


def checkpoint_tensor_bytes(directory):
    paths = sorted(directory.glob('*.safetensors'))
    if not paths:
        raise ValueError('local safetensors checkpoint required')
    total, headers = 0, {}
    for path in paths:
        with path.open('rb') as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError('invalid checkpoint header')
            header_size = struct.unpack('<Q', raw_size)[0]
            if not 0 < header_size <= min(path.stat().st_size - 8, 64 * 1024**2):
                raise ValueError('invalid checkpoint header length')
            raw_header = handle.read(header_size)
        header = json.loads(raw_header)
        headers[path.name] = hashlib.sha256(raw_header).hexdigest()
        offsets = sorted(tuple(value['data_offsets']) for name, value in header.items()
                         if name != '__metadata__')
        cursor = 0
        for begin, end in offsets:
            if begin != cursor or end < begin:
                raise ValueError('checkpoint storage offsets overlap or contain gaps')
            total += end - begin
            cursor = end
        if cursor + header_size + 8 != path.stat().st_size:
            raise ValueError('checkpoint header and payload size differ')
    return total, headers


def project(smoke, joint, *, device_capacity_bytes, os_reserve_bytes,
            runtime_reserve_bytes, pim_pool_bytes, embedder_tensor_bytes=0):
    if (device_capacity_bytes <= 0 or pim_pool_bytes <= 0
            or min(os_reserve_bytes, runtime_reserve_bytes, embedder_tensor_bytes) < 0):
        raise ValueError('capacity and reserve values must be nonnegative')
    if (not smoke.get('rows') or joint['pim_pool_capacity_bytes'] != pim_pool_bytes
            or not joint['known_joint_pool_fits']):
        raise ValueError('nonempty trace and consistent feasible shared PIM pool required')
    checkpoint = joint['reader_checkpoint_tensor_bytes']
    primary = smoke.get('reader_integer_primary')
    resident_reader = checkpoint
    if smoke.get('reader_quantization') == 'w8a8':
        if not primary:
            raise ValueError('W8A8 capacity requires the actual primary conversion inventory')
        required = ('model_footprint_after_bytes', 'integer_primary_bytes',
                    'retained_bf16_embedding_alias_bytes', 'released_bf16_parameter_bytes',
                    'replaced_source_parameter_bytes')
        if (any(type(primary.get(field)) is not int or primary[field] < 0 for field in required)
                or primary.get('source_weight_copies_retained_by_converter') is not False
                or primary['released_bf16_parameter_bytes'] + primary['retained_bf16_embedding_alias_bytes']
                   != primary['replaced_source_parameter_bytes']
                or primary['model_footprint_after_bytes'] < primary['integer_primary_bytes']
                   + primary['retained_bf16_embedding_alias_bytes']
                or any(name.startswith('hypothetical_') for name in joint['known_reservations'])):
            raise ValueError('inconsistent INT8 primary, embedding aliases or duplicate hypothetical storage')
        resident_reader = primary['model_footprint_after_bytes']
    known_state = (joint['observed_embedding_state_peak_bytes']
                   + joint['observed_visual_state_peak_bytes']
                   + joint['known_reservations']['dense_fp32_index_bytes'])
    declared_state = sum(joint['known_reservations'].values())
    if min(checkpoint, known_state, declared_state) < 0 or known_state > declared_state:
        raise ValueError('invalid observed or declared tensor residency')
    functional_layout_bytes = 0
    functional_declared_increment = 0
    functional_extra_cuda_bytes = None
    if smoke.get('reader_functional_backend', 'host_primary') != 'host_primary':
        startup = smoke.get('pim_functional_startup') or {}
        dispatch = smoke.get('pim_functional_dispatch') or {}
        functional_layout_bytes = startup.get('layout_resident_bytes')
        functional_extra_cuda_bytes = startup.get('functional_extra_cuda_bytes')
        included_padding = joint['known_reservations'].get('additional_bank_layout_padding_bytes', 0)
        if (smoke['reader_functional_backend'] != 'pimnast_bank_layout_reference'
                or startup.get('physical_primary_aliasing_verified') is not False
                or type(functional_layout_bytes) is not int
                or type(functional_extra_cuda_bytes) is not int
                or not 0 <= included_padding <= functional_layout_bytes
                or functional_extra_cuda_bytes < functional_layout_bytes
                or dispatch.get('resident_layout_bytes') != functional_layout_bytes
                or startup.get('declared_weight_pool_bytes') != pim_pool_bytes
                or functional_layout_bytes > pim_pool_bytes):
            raise ValueError('explicit duplicate functional layout and consistent pool required')
        functional_declared_increment = functional_layout_bytes - included_padding
    known_lower_bound = resident_reader + embedder_tensor_bytes + known_state + functional_layout_bytes
    declared_payload = resident_reader + embedder_tensor_bytes + declared_state + functional_declared_increment
    app_budget = device_capacity_bytes - os_reserve_bytes
    payload_budget = app_budget - runtime_reserve_bytes
    peak_cuda = max(row['service_peak_cuda_allocated_bytes'] for row in smoke['rows'])
    projected_peak = declared_payload + runtime_reserve_bytes
    pim_fits = pim_pool_bytes <= max(0, app_budget)
    return {
        'device_capacity_bytes': device_capacity_bytes,
        'os_reserve_bytes': os_reserve_bytes,
        'runtime_reserve_bytes': runtime_reserve_bytes,
        'application_budget_bytes': app_budget,
        'payload_budget_after_runtime_reserve_bytes': payload_budget,
        'pim_pool_bytes': pim_pool_bytes,
        'reader_checkpoint_tensor_bytes': checkpoint,
        'reader_resident_tensor_bytes': resident_reader,
        'functional_duplicate_layout_bytes': functional_layout_bytes,
        'functional_declared_payload_increment_bytes': functional_declared_increment,
        'observed_server_functional_extra_cuda_bytes': functional_extra_cuda_bytes,
        'pim_layout_alias_assumed': functional_layout_bytes == 0,
        'reader_storage_mode': 'measured_converted_primary_with_retained_embedding' if primary else 'checkpoint_tensor_payload',
        'observed_server_conversion_peak_cuda_allocated_bytes': primary.get('conversion_peak_cuda_allocated_bytes') if primary else None,
        'startup_capacity_qualified': False,
        'embedder_checkpoint_tensor_bytes': embedder_tensor_bytes,
        'known_materialized_state_bytes': known_state,
        'known_unique_lower_bound_bytes': known_lower_bound,
        'declared_unique_payload_bytes': declared_payload,
        'observed_server_peak_cuda_allocated_bytes': peak_cuda,
        'projected_peak_with_declared_runtime_reserve_bytes': projected_peak,
        'known_lower_bound_fits_application_budget': known_lower_bound <= payload_budget,
        'projected_peak_fits_application_budget': pim_fits and projected_peak <= app_budget,
        'reader_weights_fit_application_budget': resident_reader <= app_budget,
        'pim_pool_fits_application_budget': pim_fits,
        'capacity_scenario_qualified_only_under_declared_reserves': pim_fits and projected_peak <= app_budget,
        'full_device_residency_qualified': False,
        'pim_pool_is_subreservation_not_extra_device_memory': True,
        'boundary': 'Reader and embedder tensors plus full declared stores and explicit runtime reserve; runtime charged once; functional bank-layout duplicates are additionally charged, with already-reserved layout padding counted once; otherwise PIM aliasing is a target-model assumption, not a verified implementation; server allocator excess and emulator workspace are not calibrated target memory; no phone capacity qualification',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--joint-capacity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--os-reserve-bytes', type=int, required=True)
    parser.add_argument('--runtime-reserve-bytes', type=int, required=True)
    parser.add_argument('--pim-pool-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--device', action='append', nargs=2, metavar=('NAME', 'BYTES'),
                        required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh device-capacity projection required')
    smoke = json.loads(args.smoke.read_text())
    joint = json.loads(args.joint_capacity.read_text())
    if joint['smoke_sha256'] != digest(args.smoke):
        raise ValueError('joint capacity producer differs from Reader smoke')
    embedder = Path(smoke['embedder'])
    if digest(embedder / 'config.json') != smoke['embedder_config_sha256']:
        raise ValueError('embedder identity changed')
    embedder_bytes, embedder_headers = checkpoint_tensor_bytes(embedder)
    devices = []
    for name, raw_bytes in args.device:
        value = int(raw_bytes)
        devices.append({'name': name, **project(
            smoke, joint, device_capacity_bytes=value,
            os_reserve_bytes=args.os_reserve_bytes,
            runtime_reserve_bytes=args.runtime_reserve_bytes,
            pim_pool_bytes=args.pim_pool_bytes, embedder_tensor_bytes=embedder_bytes)})
    result = {
        'schema_version': 'omni-device-capacity-projection-v2',
        'smoke_sha256': digest(args.smoke),
        'joint_capacity_sha256': digest(args.joint_capacity),
        'devices': devices,
        'embedder_tensor_header_sha256': embedder_headers,
        'script_sha256': digest(Path(__file__)),
        'assumptions': {
            'os_reserve_bytes': args.os_reserve_bytes,
            'runtime_reserve_bytes': args.runtime_reserve_bytes,
            'pim_pool_bytes': args.pim_pool_bytes,
        },
        'boundary': 'server-derived capacity projection; no phone deployment, OS memory trace, quantized checkpoint, or PIM silicon measurement',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
