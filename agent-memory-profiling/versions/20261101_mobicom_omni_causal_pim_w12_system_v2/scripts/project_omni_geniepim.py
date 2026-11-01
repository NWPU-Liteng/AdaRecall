"""Run pinned upstream LPDDR-PIM GEMV costs; never substitute them for full-service latency."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.decode_gemv import select_exact_capacity
from project_omni_device_capacity import checkpoint_tensor_bytes, project as project_device_capacity
from omni_decode_lifecycle import same_platform_linear_lifecycle

UPSTREAM_COMMIT = 'c282f0159f493b0c627b14ab28adfdb957c94692'
UPSTREAM_TREE = 'b3b83744ce8d74244bae523b95580e477baa2344'
ADAPTATIONS = {
    'HOST_PEAK_MEM_COMPUTE_LIST': 'IPU/MEM:120:0.75/4:66.352:1&6:44.235:1&8:33.176:1&16:16.588:1&32:8.294:1',
    'OPERAND_SIZE_LIST': '16',
    'ACT_OPERAND_SIZE_LIST': '16',
    'IGNORE_HOST_READ_INPUT_FLAG_FOR_HOST_EXEC_LIST': 'False',
    'IGNORE_HOST_WRITE_OUTPUT_FLAG_FOR_HOST_EXEC_LIST': 'False',
    'IGNORE_OUTPUT_REGISTER_PRESSURE_FLAG_LIST': 'False',
    'IGNORE_INPUT_REGISTER_PRESSURE_FLAG_LIST': 'False',
    'ASSUME_FREE_CROSS_SIMD_REDUCTION_LIST': 'False',
    'PIM_TILE_SHAPE_DEGREE_LIST': '16',
    'PIM_TILE_ORDER_DEGREE_LIST': '4',
}


def config_adaptations(operand_bits):
    if operand_bits not in (8, 16):
        raise ValueError('explicit 8-bit or 16-bit scalar operand scenario required')
    return dict(ADAPTATIONS, OPERAND_SIZE_LIST=str(operand_bits),
                ACT_OPERAND_SIZE_LIST=str(operand_bits),
                PIM_TILE_ORDER_DEGREE_LIST='1' if operand_bits == 8 else '4')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_commit(path):
    try:
        return subprocess.check_output(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_id(path):
    def object_id(kind, data):
        return hashlib.sha1(kind + b' ' + str(len(data)).encode() + b'\0' + data).digest()

    entries = []
    for item in path.iterdir():
        if item.name in {'__pycache__', '.git', 'Outputs'}:
            continue
        if item.is_symlink():
            mode, identity, key = b'120000', object_id(b'blob', os.readlink(item).encode()), item.name
        elif item.is_dir():
            mode, identity, key = b'40000', bytes.fromhex(source_tree_id(item)), item.name + '/'
        else:
            mode = b'100755' if item.stat().st_mode & 0o111 else b'100644'
            identity, key = object_id(b'blob', item.read_bytes()), item.name
        entries.append((key, mode + b' ' + item.name.encode() + b'\0' + identity))
    return object_id(b'tree', b''.join(record for _, record in sorted(entries))).hex()


def adapt_config(text, operand_bits=16):
    adaptations = config_adaptations(operand_bits)
    remaining = set(adaptations)
    lines = []
    for line in text.splitlines():
        key = line.split('=', 1)[0]
        if key in adaptations:
            line = key + '=' + adaptations[key]
            remaining.remove(key)
        lines.append(line)
    if remaining:
        raise ValueError('upstream configuration changed: ' + str(sorted(remaining)))
    return '\n'.join(lines) + '\n'


def residual_budget(host_ns, hybrid_ns, target_speedup):
    if (not all(math.isfinite(value) and value >= 0 for value in (host_ns, hybrid_ns))
            or not math.isfinite(target_speedup) or target_speedup <= 1):
        raise ValueError('finite nonnegative costs and speedup above one required')
    value = (host_ns - target_speedup * hybrid_ns) / (target_speedup - 1)
    return value if value >= 0 else None


def integer_output_cost(cost, outputs, inputs, operand_bits):
    host_ns = cost['host_gemm_time_ns']
    if operand_bits != 8:
        return host_ns, None
    expected_bytes = outputs * inputs + inputs + outputs
    if not math.isclose(cost['host_memory_bytes'], expected_bytes):
        raise ValueError('scalar INT8 matrix/input/output upstream memory contract changed')
    corrected_bytes = expected_bytes + 3 * outputs
    corrected_ns = max(cost['host_compute_time_ns'], corrected_bytes / 90.)
    if cost['pim_host_read_output_bytes'] < 4 * outputs:
        raise ValueError('PIM must return at least the complete INT32 output vector')
    return corrected_ns, {
        'primitive': 'INT8 x INT8 -> INT32, before FP32 dequantization and BF16 conversion',
        'upstream_host_output_bytes': outputs,
        'integer_host_output_bytes': 4 * outputs,
        'host_memory_bytes': corrected_bytes,
        'host_bandwidth_bytes_per_ns': 90.,
        'upstream_host_per_call_ns': host_ns,
        'pim_output_return_bytes_already_charged': cost['pim_host_read_output_bytes'],
        'extra_pim_output_return_charge_bytes': 0,
        'activation_quantization_cost_ns': None,
        'fp32_scale_bias_and_bf16_conversion_cost_ns': None,
        'boundary': 'corrects the raw integer primitive only on pinned same-platform MEM sources; padded INT32 PIM output already included upstream; tensor footprint is not a measured DRAM-traffic claim'}


def capacity_selection(candidates, capacity_bytes, *, group_gate_up):
    groups = {}
    for name, item in candidates.items():
        if (group_gate_up and '.mlp.' in name
                and name.rsplit('.', 1)[-1] in {'gate_proj', 'up_proj'}):
            group = name.rsplit('.', 1)[0] + '.gate_up'
        else:
            group = name
        groups.setdefault(group, []).append(name)
    costs = {}
    for group, names in groups.items():
        if group.endswith('.gate_up'):
            if {name.rsplit('.', 1)[-1] for name in names} != {'gate_proj', 'up_proj'}:
                raise ValueError('complete observed gate/up siblings required')
            if len({(candidates[name]['shape'], candidates[name]['calls']) for name in names}) != 1:
                raise ValueError('gate/up must have matching shapes and executing call counts')
        host_ns = sum(candidates[name]['host_per_call_ns'] for name in names)
        pim_ns = sum(candidates[name]['pim_per_call_ns'] for name in names)
        size = sum(candidates[name]['resident_bytes'] for name in names)
        costs[group] = {'resident_bytes': size,
                        'host_cycles': math.floor(host_ns * 1000),
                        'pim_cycles': math.ceil(pim_ns * 1000),
                        'capacity_feasible': size <= capacity_bytes, 'buffer_feasible': True}
    selected_groups, used, gain = select_exact_capacity(costs, capacity_bytes)
    selected = {name for group in selected_groups for name in groups[group]}
    return selected, used, {'selected_groups': sorted(selected_groups),
                           'groups': groups, 'per_call_gain_picoseconds': gain,
                           'optimizer': 'existing pim_backend.decode_gemv.select_exact_capacity',
                           'future_call_count_used_by_objective': False,
                           'gate_up_policy': 'co-resident siblings; no free broadcast/reduction/fence elimination credited',
                           'surrogate_units': 'integer picoseconds; not DRAM clock cycles'}


def worker(runtime, output):
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    sys.argv = ['run_geniepim_core.py', '--verbose', '0']
    from geniepim_c_combinations_generator import COMBINATIONS
    from gemm_generator import GEMM_LIST
    from geniepim_core import geniepim_core
    if len(COMBINATIONS) != 1:
        raise ValueError('one frozen same-platform configuration required')
    rows = []
    for matrix in GEMM_LIST:
        instance = geniepim_core(matrix, COMBINATIONS[0])
        instance.run_geniepim_core()
        rows.append({'matrix_id': instance.host_gemm.gemm_source_id,
                     'out_features': instance.host_gemm.gemm_m,
                     'in_features': instance.host_gemm.gemm_k,
                     'cost': vars(instance.output)})
    with output.open('x') as handle:
        json.dump(rows, handle, indent=2)


def project(runs, costs, operand_bits=16, pim_capacity_bytes=None):
    if operand_bits not in (8, 16):
        raise ValueError('8-bit or 16-bit operand scenario required')
    costs = {row['matrix_id']: row for row in costs}
    default_capacity_bytes = 2 * 1024**3
    capacity_bytes = (default_capacity_bytes if pim_capacity_bytes is None
                      else pim_capacity_bytes)
    if type(capacity_bytes) is not int or capacity_bytes <= 0:
        raise ValueError('PIM capacity must be a positive integer byte count')
    users = []
    checkpoint_sizes = {}
    for path, run in runs:
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError('complete label-isolated generation required')
        records = [row for row in run['operator_trace']['operators'] if row['phase'] == 'decode']
        element_bytes = 1 if run.get('reader_quantization') == 'w8a8' else 2
        if (not records or any(row['rows'] != 1 or row['element_bytes'] != element_bytes
                               or row['weight_bytes'] != row['in_features'] * row['out_features'] * element_bytes
                               for row in records)
                or (element_bytes == 1 and operand_bits != 8)):
            raise ValueError('consistent one-row decode trace and matching integer operand scenario required')
        host_ns = hybrid_ns = placement_ns = 0.
        matrices = []
        candidates = {}
        dense_weights = {}
        for row in records:
            shape = f"{row['out_features']}x{row['in_features']}"
            cost = costs[shape]['cost']
            host, output_contract = integer_output_cost(
                cost, row['out_features'], row['in_features'], operand_bits)
            pim = cost['pim_time_ns']
            rows_per_bank = max(1, math.ceil(cost['pim_matrix_scalar_num_occupied_dram_rows_per_bank']))
            padded_bytes = rows_per_bank * 2048 * 128
            candidates[row['name']] = {'resident_bytes': padded_bytes,
                                       'host_per_call_ns': host, 'pim_per_call_ns': pim,
                                       'calls': row['calls'], 'shape': shape,
                                       'integer_output_cost_contract': output_contract}
            dense_weights[row['name']] = row['weight_bytes'] * operand_bits // (element_bytes * 8)
            host_ns += host * row['calls']
        individual, individual_bytes, individual_policy = capacity_selection(
            candidates, capacity_bytes, group_gate_up=False)
        selected, resident_bytes, group_policy = capacity_selection(
            candidates, capacity_bytes, group_gate_up=True)
        independent_hybrid_ns = 0.
        individual_placement_ns = 0.
        for name, item in candidates.items():
            if name in individual:
                independent_hybrid_ns += item['pim_per_call_ns'] * item['calls']
                individual_placement_ns += item['resident_bytes'] / (120e9 * 0.75) * 1e9
            else:
                independent_hybrid_ns += item['host_per_call_ns'] * item['calls']
        for name, item in candidates.items():
            selected_now = name in selected
            hybrid_ns += (item['pim_per_call_ns'] if selected_now else item['host_per_call_ns']) * item['calls']
            if selected_now:
                placement_ns += item['resident_bytes'] / (120e9 * 0.75) * 1e9
            matrices.append({'name': name, 'shape': item['shape'], 'calls': item['calls'],
                             'model_route': 'pim' if selected_now else 'host',
                             'padded_resident_bytes': item['resident_bytes'] if selected_now else 0,
                             'candidate_resident_bytes': item['resident_bytes'],
                             'host_per_call_ns': item['host_per_call_ns'],
                             'pim_per_call_ns': item['pim_per_call_ns'],
                             'integer_output_cost_contract': item['integer_output_cost_contract']})
        for role in ('reader', 'embedder'):
            directory = Path(run[role])
            if directory not in checkpoint_sizes:
                checkpoint_sizes[directory] = checkpoint_tensor_bytes(directory)
        reader_bytes, reader_headers = checkpoint_sizes[Path(run['reader'])]
        embedder_bytes, embedder_headers = checkpoint_sizes[Path(run['embedder'])]
        layout_padding = resident_bytes - sum(dense_weights[name] for name in selected)
        if layout_padding < 0:
            raise ValueError('bank resident layout smaller than its actual dense weights')
        embedder_config = Path(run['embedder']) / 'config.json'
        if digest(embedder_config) != run['embedder_config_sha256']:
            raise ValueError('embedding shape configuration changed')
        dimension = json.loads(embedder_config.read_text())['hidden_size']
        dense_index = run['embedding_windows'] * dimension * 4
        store = run['embedding_state_reuse']
        visual = run.get('visual_state_reuse') or {}
        capacity_inventory = {
            'pim_pool_capacity_bytes': capacity_bytes,
            'profile_default_pim_pool_capacity_bytes': default_capacity_bytes,
            'pim_capacity_override_bytes': pim_capacity_bytes,
            'pim_capacity_override_applied': pim_capacity_bytes is not None,
            'known_joint_pool_fits': resident_bytes <= capacity_bytes,
            'reader_checkpoint_tensor_bytes': reader_bytes,
            'observed_embedding_state_peak_bytes': store['peak_resident_bytes'],
            'observed_visual_state_peak_bytes': visual.get('peak_resident_bytes', 0),
            'known_reservations': {'dense_fp32_index_bytes': dense_index,
                'embedding_store_bytes': store['capacity_bytes'] if run['embedding_state_reuse_enabled'] else 0,
                'visual_store_bytes': visual.get('capacity_bytes', 0),
                'additional_bank_layout_padding_bytes': layout_padding}}
        if operand_bits == 8 and element_bytes == 2:
            capacity_inventory['known_reservations']['hypothetical_int8_primary_bytes'] = sum(dense_weights.values())
            capacity_inventory['known_reservations']['hypothetical_fp32_row_scale_bytes'] = sum(
                row['out_features'] * 4 for row in records)
        device_capacity = project_device_capacity(run, capacity_inventory,
            device_capacity_bytes=12 * 1024**3, os_reserve_bytes=1024**3,
            runtime_reserve_bytes=1024**3, pim_pool_bytes=capacity_bytes,
            embedder_tensor_bytes=embedder_bytes)
        users.append({'run': str(path), 'run_sha256': digest(path), 'known_linear_host_ns': host_ns,
                      'known_linear_hybrid_ns': hybrid_ns,
                      'cold_layout_write_ns_lower_bound': placement_ns,
                      'known_linear_warm_ratio': host_ns / hybrid_ns,
                      'independent_matrix_placement_host_ns': host_ns,
                      'independent_matrix_placement_hybrid_ns': independent_hybrid_ns,
                      'independent_matrix_placement_speedup': host_ns / independent_hybrid_ns,
                      'joint_gate_up_placement_speedup': host_ns / hybrid_ns,
                      'joint_grouping_warm_cost_regret_ns': hybrid_ns - independent_hybrid_ns,
                      'independent_matrix_placement_cold_minimum_ns': individual_placement_ns,
                      'joint_gate_up_placement_cold_minimum_ns': placement_ns,
                      'physical_padded_weight_bytes_without_alias_dedup': resident_bytes,
                      'declared_pim_pool_bytes': capacity_bytes,
                      'profile_default_pim_pool_bytes': default_capacity_bytes,
                      'pim_capacity_override_bytes': pim_capacity_bytes,
                      'unconstrained_positive_gain_bytes': sum(item['resident_bytes'] for item in candidates.values()
                                                              if item['pim_per_call_ns'] < item['host_per_call_ns']),
                      'selected_weights_fit_declared_pool': resident_bytes <= capacity_bytes,
                      'capacity_constrained_route_qualified': False,
                      'group_residency_policy': group_policy,
                      'whole_device_capacity_scenario': device_capacity,
                      'integer_storage_boundary': ('actual converted INT8 primary, scales, non-Linear BF16 tensors and tied BF16 embedding charged by measured model footprint; checkpoint is not a second resident copy; conversion peak and unqualified startup remain visible'
                                                   if element_bytes == 1 else
                                                   'BF16 checkpoint retained; full hypothetical INT8 Linear primary and FP32 row scales charged additionally; selected PIM layout aliases only that integer primary; no unimplemented BF16 removal credited'
                                                   if operand_bits == 8 else None),
                      'declared_reserve_boundary': '12 GiB hypothetical total, 1 GiB OS and 1 GiB runtime assumptions, not phone measurements; PIM pool aliases primary weights; extra layout padding charged once',
                      'reader_safetensors_header_sha256': reader_headers,
                      'embedder_safetensors_header_sha256': embedder_headers,
                      'individual_control': {'resident_bytes': individual_bytes,
                          'selected_names': sorted(individual),
                          'known_linear_hybrid_ns': independent_hybrid_ns,
                          'policy': individual_policy},
                      'model_time_units_comparable_to_measured_server_wall_clock': False,
                      'known_linear_request_lifecycle': same_platform_linear_lifecycle(
                          run, candidates, selected, placement_ns),
                      'maximum_unmapped_residual_ns_for_1_3x_warm': residual_budget(host_ns, hybrid_ns, 1.3),
                      'maximum_unmapped_residual_ns_for_1_3x_cold': residual_budget(host_ns, hybrid_ns + placement_ns, 1.3),
                      'matrices': matrices})
    return {'complete': True, 'per_user': users,
            'operand_bits': operand_bits,
            'architecture_reference': f'AMD GeniePIM upstream config.in, one stack × 8 channels × 16 banks, LPDDR5X timing tuple, IPU host 120 GB/s at configured 75% utilization, {operand_bits}-bit scalar operands',
            'full_model_host_seconds': None, 'full_model_hybrid_seconds': None,
            'full_system_speedup': None, 'functional_generation_quality_qualified': False,
            'capacity_qualified': False, 'pim_physical_device_measured': False,
            'boundary': 'capacity-constrained shape scenario, not a functional or fused scheduler; all unmapped work including quantization/scaling remains unknown, never zero; costs do not certify arithmetic or quality; cold cost is only minimum layout write; whole-device memory/identity qualification remains required; INT8 scenario changes both Host and PIM operands, not a speedup over BF16 Host'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--runs', type=Path, nargs='+')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--worker-runtime', type=Path)
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--operand-bits', type=int, choices=(8, 16), default=16)
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument('--pim-capacity-bytes', type=int)
    capacity_group.add_argument('--pim-capacity-gib', type=float)
    args = parser.parse_args()
    if args.worker_runtime:
        worker(args.worker_runtime.resolve(), args.worker_output.resolve())
        return
    if not args.upstream or not args.runs or not args.output_root or args.output_root.exists():
        raise ValueError('upstream, complete runs and fresh output directory required')
    root = args.output_root.resolve()
    runtime = root / 'runtime'
    upstream = args.upstream.resolve()
    tree_identity = source_tree_id(upstream)
    if tree_identity != UPSTREAM_TREE:
        raise ValueError('upstream source tree differs from the pinned official commit')
    root.mkdir(parents=True)
    source_snapshot = root / 'sources'
    integration_sources = [Path(__file__).resolve(), ROOT / 'pim_backend/decode_gemv.py',
                           ROOT / 'scripts/project_omni_device_capacity.py',
                           ROOT / 'scripts/omni_decode_lifecycle.py']
    integration_hashes = {}
    for path in integration_sources:
        relative = path.relative_to(ROOT)
        target = source_snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        integration_hashes[str(relative)] = digest(target)
        if integration_hashes[str(relative)] != digest(path):
            raise ValueError('integration source changed while freezing projection')
    hashes = {str(path.relative_to(upstream)): digest(path) for path in upstream.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts}
    shutil.copytree(upstream, runtime, ignore=shutil.ignore_patterns('__pycache__', 'Outputs'))
    runs = [(path, json.loads(path.read_text())) for path in args.runs]
    shapes = sorted({(row['out_features'], row['in_features']) for _, run in runs
                     for row in run['operator_trace']['operators'] if row['phase'] == 'decode'})
    config = upstream / 'Inputs/Configs/config.in'
    (runtime / 'Inputs/Configs/config.in').write_text(adapt_config(config.read_text(), args.operand_bits))
    with (runtime / 'Inputs/GEMMs/gemm.in').open('w') as handle:
        handle.write('name,matrix_id,gemm_m,gemm_k,batch_size,gemm_n_values\n')
        for outputs, inputs in shapes:
            handle.write(f'Qwen3.5-4B,{outputs}x{inputs},{outputs},{inputs},1,1\n')
    registration = {'upstream_commit': UPSTREAM_COMMIT,
                    'upstream_commit_provenance': 'full Git tree object hash matches official AMDGeniePIM commit tree; source may be archive rather than nested checkout',
                    'upstream_tree_sha1': tree_identity,
                    'source_checkout_repository_commit': repository_commit(upstream),
                    'integration_source_snapshot_directory': str(source_snapshot),
                    'integration_sources_sha256': integration_hashes,
                    'upstream_sources_sha256': hashes,
                    'official_reference': 'PIMnast SC24-W DOI 10.1109/SCW63240.2024.00137; AMD GeniePIM v0.7.0',
                    'config_source_sha256': digest(config), 'adaptations': config_adaptations(args.operand_bits),
                    'generated_config_sha256': digest(runtime / 'Inputs/Configs/config.in'),
                    'script_sha256': digest(Path(__file__)),
                    'existing_allocator_sha256': digest(ROOT / 'pim_backend/decode_gemv.py'),
                    'existing_device_capacity_sha256': digest(ROOT / 'scripts/project_omni_device_capacity.py'),
                    'existing_decode_lifecycle_sha256': digest(ROOT / 'scripts/omni_decode_lifecycle.py'),
                    'unknowns': ['phone calibration', 'full functional hardware mapping', 'runtime capacity',
                                 'layout construction', 'group boundaries/fences', 'opaque kernel timing',
                                 'dynamic activation quantization and FP32 scale/bias/cast timing']}
    (root / 'registration.json').write_text(json.dumps(registration, indent=2))
    costs = root / 'matrix_costs.json'
    with (root / 'model.log').open('x') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-runtime', str(runtime),
                        '--worker-output', str(costs)], stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.pim_capacity_gib is not None:
        if args.pim_capacity_gib <= 0:
            raise ValueError('PIM capacity in GiB must be positive')
        capacity_override = int(args.pim_capacity_gib * 1024**3)
    else:
        capacity_override = args.pim_capacity_bytes
    result = project(runs, json.loads(costs.read_text()), args.operand_bits, capacity_override)
    if any(digest(ROOT / name) != expected for name, expected in integration_hashes.items()):
        raise ValueError('projection integration source changed during execution')
    result['registration_sha256'] = digest(root / 'registration.json')
    result['matrix_costs_sha256'] = digest(costs)
    (root / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('complete', 'full_system_speedup', 'capacity_qualified')}))


if __name__ == '__main__':
    main()
