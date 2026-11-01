#!/usr/bin/env python3
"""Qualify a declared shared-vector traffic contract, not LPDDR-PIM hardware."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT.parents[2] / 'attacc_simulator-main/ramulator2'
POOL_BYTES = 2 * 1024**3
TRANSFER_BYTES = 32
NATIVE_SOURCES = (
    'base/factory.cpp', 'base/logging.cpp', 'base/request.cpp', 'base/stats.cpp',
    'base/utils.cpp', 'base/config.cpp', 'dram/impl/LPDDR5.cpp',
    'memory_system/impl/generic_DRAM_system.cpp',
    'dram_controller/impl/generic_dram_controller.cpp',
    'dram_controller/impl/scheduler/generic_scheduler.cpp',
    'dram_controller/impl/refresh/all_bank_refresh.cpp',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rd16_coordinates(address):
    if address < 0 or address >= 2**32 or address % TRANSFER_BYTES:
        raise ValueError('unaligned or out-of-capacity RD16 address')
    transaction = address // TRANSFER_BYTES
    channel = transaction & 1
    column = ((transaction >> 1) & 63) * 16
    bankgroup = (transaction >> 7) & 3
    bank = (transaction >> 9) & 3
    row = transaction >> 11
    return [channel, 0, bankgroup, bank, row, column]


def make_config(command_path):
    return {'MemorySystem': {
        'impl': 'GenericDRAM', 'clock_ratio': 1,
        'DRAM': {'impl': 'LPDDR5',
                 'org': {'preset': 'LPDDR5_16Gb_x16', 'channel': 2, 'channel_width': 16},
                 'timing': {'preset': 'LPDDR5_6400'}},
        'Controller': {'impl': 'Generic', 'Scheduler': {'impl': 'FRFCFS'},
                       'RefreshManager': {'impl': 'AllBank'},
                       'plugins': [{'ControllerPlugin': {
                           'impl': 'MaterializationCommandAudit', 'path': str(command_path.resolve())}}]},
        'AddrMapper': {'impl': 'LPDDR5RD16'},
    }}


def emit_shared_traffic(atoms, rows, dimension, output):
    if dimension < 1 or dimension % TRANSFER_BYTES:
        raise ValueError('dimension must be a positive multiple of 32')
    if len({atom['identity'] for atom in atoms}) != len(atoms):
        raise ValueError('immutable identities are not distinct')
    if len(atoms) * dimension > POOL_BYTES:
        raise ValueError('shared vector pool capacity exceeded')
    if not rows or len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError('non-empty distinct request stream required')
    if [row['effective_timestamp'] for row in rows] != sorted(row['effective_timestamp'] for row in rows):
        raise ValueError('request stream is not chronological')
    if any(row['split'] != 'development' or row['status'] != 'ok' for row in rows):
        raise ValueError('only completed development requests are permitted')
    if any(row['feature_cache']['evictions'] for row in rows):
        raise ValueError('this qualification requires a no-eviction source stream')
    materialized, phases, inventory = set(), [], []
    query_address = POOL_BYTES
    output_address = query_address + dimension
    counts, bank_counts = Counter(), Counter()
    channel_counts = {str(channel): Counter() for channel in (0, 1)}

    def phase(handle, question_id, name, ranges):
        phase_counts = Counter()
        for operation, start, size in ranges:
            if start % TRANSFER_BYTES or size < 0 or size % TRANSFER_BYTES:
                raise ValueError('unaligned transfer range')
            for address in range(start, start + size, TRANSFER_BYTES):
                coordinates = rd16_coordinates(address)
                handle.write(f'{operation} {address}\n')
                phase_counts[operation] += 1
                bank_counts[f'{coordinates[0]}/{coordinates[2]}/{coordinates[3]}'] += 1
                channel_counts[str(coordinates[0])][operation] += 1
        handle.write(f'FENCE {len(phases)}\n')
        counts.update(phase_counts)
        phases.append({'question_id': question_id, 'name': name,
                       'read_requests': phase_counts['LD'], 'write_requests': phase_counts['ST']})

    with output.open('w') as handle:
        for row in rows:
            visible = [index for index, atom in enumerate(atoms)
                       if atom['timestamp'] <= row['effective_timestamp']]
            if not visible or len(visible) != row['visible_atoms']:
                raise ValueError('source visible-atom count differs from regenerated atoms')
            new = [index for index in visible if index not in materialized]
            if len(materialized) + len(new) != row['feature_cache']['misses']:
                raise ValueError('source feature-cache cumulative misses do not match immutable producers')
            ranges = [('ST', index * dimension, dimension) for index in new]
            ranges.append(('ST', query_address, dimension))
            phase(handle, row['question_id'], 'producer_commit', ranges)
            for index in new:
                atom = atoms[index]
                inventory.append({'identity': atom['identity'], 'event_id': atom['event_id'],
                                  'timestamp': atom['timestamp'], 'address': index * dimension,
                                  'bytes': dimension, 'first_request': row['question_id']})
            materialized.update(new)
            ranges = [('LD', query_address, dimension)]
            ranges += [('LD', index * dimension, dimension) for index in visible]
            phase(handle, row['question_id'], 'host_scan_inputs', ranges)
            top_k = min(128, len(visible))
            result_bytes = ((top_k * 8 + TRANSFER_BYTES - 1) // TRANSFER_BYTES) * TRANSFER_BYTES
            phase(handle, row['question_id'], 'topk_result_commit', [('ST', output_address, result_bytes)])
    return {'traffic_contract': 'producer_once_shared_int8_scan_v1',
            'requests': len(rows), 'unique_vectors': len(materialized),
            'dimension': dimension, 'transfer_bytes': TRANSFER_BYTES,
            'producer_vector_bytes': len(materialized) * dimension,
            'producer_query_bytes': len(rows) * dimension,
            'read_requests': counts['LD'], 'write_requests': counts['ST'],
            'transferred_bytes': (counts['LD'] + counts['ST']) * TRANSFER_BYTES,
            'vector_pool_capacity_bytes': POOL_BYTES,
            'active_bytes': len(materialized) * dimension + dimension + 128 * 8,
            'phases': phases, 'bank_requests': dict(bank_counts), 'inventory': inventory,
            'channel_requests': {channel: dict(counts) for channel, counts in channel_counts.items()},
            'representations': 'one primary INT8 vector per immutable atom; query/output scratch reused after fences',
            'omitted_work': ['embedding/quantization compute', 'FP32 feature pool and cosine rescoring',
                             'host scoring/topk compute', 'reranker and Reader', 'CPU/GPU caches',
                             'request arrival idle gaps and full-model contention'],
            'boundary': 'proposed shared INT8 memory-side traffic contract, not captured current-service DRAM traffic; no PIM MACs, value simulation, or device calibration'}


def qualify(traffic, report, channels):
    expected_reads = traffic['read_requests']
    expected_writes = traffic['write_requests']
    command_totals = Counter()
    channel_ids = [channel['channel'] for channel in channels]
    for channel in channels:
        command_totals.update(channel['commands'])
    fences = report.get('fences', [])
    previous_reads = previous_writes = previous_cycle = 0
    phases_match = len(fences) == len(traffic['phases'])
    for index, (fence, phase) in enumerate(zip(fences, traffic['phases'])):
        phases_match &= (fence['phase'] == index and not fence['pending']
                         and fence['completed_reads'] == fence['accepted_reads']
                         and fence['accepted_reads'] - previous_reads == phase['read_requests']
                         and fence['accepted_writes'] - previous_writes == phase['write_requests']
                         and fence['cycle'] > previous_cycle
                         and fence['quiet_tail_cycles'] >= report['write_refresh_tail_cycles'])
        previous_reads, previous_writes, previous_cycle = (
            fence['accepted_reads'], fence['accepted_writes'], fence['cycle'])
    checks = {
        'driver_complete': report.get('complete') is True,
        'all_requests_accepted': report['read_requests'] == expected_reads
                                 and report['write_requests'] == expected_writes,
        'read_callbacks_complete': report['completed_reads'] == expected_reads,
        'all_final_commands_issued': command_totals['RD16'] == expected_reads
                                    and command_totals['WR16'] == expected_writes,
        'no_pending_at_finish': report['pending_at_finish'] is False,
        'phase_fences_and_tail_match': bool(phases_match),
        'both_channel_reports': sorted(channel_ids) == [0, 1],
        'per_channel_final_commands_match': all(
            channel['commands'].get('RD16', 0) == traffic['channel_requests'][str(channel['channel'])].get('LD', 0)
            and channel['commands'].get('WR16', 0) == traffic['channel_requests'][str(channel['channel'])].get('ST', 0)
            for channel in channels),
        'byte_address_bank_distribution_matches': report['bank_requests'] == traffic['bank_requests'],
        'rd16_mapping_matches': bool(report.get('mapping_samples')) and all(
            row['coordinates'] == rd16_coordinates(row['address']) for row in report['mapping_samples']),
        'fixed_ck_period': report['tck_ns'] == 1.25,
        'refresh_seen_on_long_trace': report['cycles'] <= 3125 or command_totals['REFab'] > 0,
    }
    return {'host_trace_qualified': all(checks.values()), 'checks': checks,
            'cycles': report['cycles'], 'modeled_memory_seconds': report['cycles'] * report['tck_ns'] * 1e-9,
            'commands': dict(command_totals), 'pim_backend_qualified': False,
            'device_calibrated': False, 'allows_development128': False,
            'boundary': 'qualified serialized memory-only LPDDR5 host trace; not compute/PIM, measured mobile latency, throughput, energy, or E2E speedup'}


def build_driver(simulator, spdlog, output):
    sources = [simulator / 'src' / name for name in NATIVE_SOURCES]
    sources.append(ROOT / 'scripts/lpddr_trace_driver.cpp')
    headers = list((simulator / 'src').rglob('*.h'))
    headers += list((simulator / 'ext/yaml-cpp/include').rglob('*.h'))
    headers += list((spdlog / 'include').rglob('*.h'))
    yaml_library = simulator / 'build/_deps/yaml-cpp-build/libyaml-cpp.a'
    if not (spdlog / 'include/spdlog/spdlog.h').is_file():
        raise ValueError('provide the pinned spdlog v1.11.0 source tree')
    version = (spdlog / 'include/spdlog/version.h').read_text()
    if any(f'#define SPDLOG_VER_{key} {value}' not in version
           for key, value in (('MAJOR', 1), ('MINOR', 11), ('PATCH', 0))):
        raise ValueError('the registered dependency version is spdlog 1.11.0')
    protocol = ROOT / 'reports/LPDDR_TRACE_CONTRACT_20261019_ZH.md'
    frozen = {str(path.resolve()): digest(path)
              for path in sources + headers + [yaml_library, Path(__file__), protocol]}
    output.mkdir()
    for path in sources + headers + [yaml_library, Path(__file__), protocol]:
        if path.is_relative_to(simulator):
            relative = Path('simulator') / path.relative_to(simulator)
        elif path.is_relative_to(spdlog):
            relative = Path('spdlog-1.11.0') / path.relative_to(spdlog)
        else:
            relative = Path('experiment') / path.relative_to(ROOT)
        target = output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    command = ['c++', '-O2', '-std=c++20', '-pthread', '-I' + str(simulator / 'src'),
               '-I' + str(simulator / 'ext/yaml-cpp/include'), '-I' + str(spdlog / 'include'),
               *map(str, sources), str(yaml_library), '-o', str(output / 'driver')]
    (output / 'build_command.json').write_text(json.dumps(command, indent=2))
    (output / 'SOURCE_MANIFEST.json').write_text(json.dumps(frozen, indent=2))
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return frozen


def run_native(directory, driver, traffic):
    import yaml
    config_path = directory / 'config.json'
    config_path.write_text(json.dumps(make_config(directory / 'commands'), indent=2))
    with (directory / 'driver.log').open('w') as log:
        subprocess.run([str(driver), str(config_path), str(directory / 'input.trace'),
                        str(directory / 'driver.yaml')], check=True, stdout=log,
                       stderr=subprocess.STDOUT, timeout=900)
    report = yaml.safe_load((directory / 'driver.yaml').read_text())
    channels = [yaml.safe_load((directory / f'commands.ch{channel}.yaml').read_text()) for channel in (0, 1)]
    return qualify(traffic, report, channels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--simulator', type=Path, default=SIMULATOR)
    parser.add_argument('--spdlog-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh qualification root required')
    result = json.loads(args.result.read_text())
    rows = result['rows']
    if (not result.get('complete') or result['configuration']['shard_count'] != 1
            or result['selected_question_ids'] != [row['question_id'] for row in rows]):
        raise ValueError('complete aligned single-stream source required')
    audit_path = args.prepared / 'audit.json'
    if digest(audit_path) != result['provenance']['prepared_audit_sha256']:
        raise ValueError('prepared audit differs from source execution')
    audit = json.loads(audit_path.read_text())
    for name in ('mobilemem_graph.json', 'manifests.jsonl'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared content differs from preparation audit')
    manifests = {row['question_id']: row for row in map(json.loads,
                 (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    for row in rows:
        manifest = manifests[row['question_id']]
        if any(row[key] != manifest[key] for key in ('person_id', 'effective_timestamp', 'split')):
            raise ValueError('request identity differs from prepared development manifest')
    if len({row['person_id'] for row in rows}) != 1:
        raise ValueError('one persistent person required')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_mobilemem_text import atomize, READER, EMBEDDER
    from transformers import AutoProcessor
    for name in ('run_mobilemem_text.py', 'mobilemem_runtime.py'):
        if digest(ROOT / 'scripts' / name) != result['provenance']['code_sha256'][name]:
            raise ValueError('atomization/runtime source changed')
    if digest(EMBEDDER / 'config.json') != result['provenance']['embedding_weights']['config.json']:
        raise ValueError('embedding dimension source changed')
    dimension = json.loads((EMBEDDER / 'config.json').read_text())['hidden_size']
    tokenizer = AutoProcessor.from_pretrained(str(READER), local_files_only=True).tokenizer
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
        if digest(READER / name) != result['provenance']['reader_weights'][name]:
            raise ValueError('atomization tokenizer changed')
    graph = next(graph for graph in json.loads((args.prepared / 'mobilemem_graph.json').read_text())
                 if graph['person_id'] == rows[0]['person_id'])
    atoms, _ = atomize(graph, tokenizer)
    args.output.mkdir(parents=True)
    traffic = emit_shared_traffic(atoms, rows, dimension, args.output / 'input.trace')
    (args.output / 'traffic.json').write_text(json.dumps(traffic, ensure_ascii=False, indent=2))
    input_paths = [args.result, audit_path, args.prepared / 'mobilemem_graph.json',
                   args.prepared / 'manifests.jsonl',
                   *[READER / name for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')],
                   EMBEDDER / 'config.json', args.output / 'input.trace', args.output / 'traffic.json',
                   ROOT / 'scripts/run_mobilemem_text.py', ROOT / 'scripts/mobilemem_runtime.py']
    frozen_inputs = {str(path.resolve()): digest(path) for path in input_paths}
    (args.output / 'registration.json').write_text(json.dumps({
        'complete': False, 'source_question_ids': result['selected_question_ids'],
        'traffic_contract': traffic['traffic_contract'], 'source_input_sha256': frozen_inputs,
        'heldout_execution_authorized': False, 'publication_ready': False}, indent=2))
    print(json.dumps({'stage': 'trace_built', 'requests': len(rows),
                      'read_requests': traffic['read_requests'], 'write_requests': traffic['write_requests']}), flush=True)
    frozen_sources = build_driver(args.simulator.resolve(), args.spdlog_source.resolve(), args.output / 'native')
    smoke = args.output / 'smoke'
    smoke.mkdir()
    smoke_atoms = [{'identity': f'vector-{index}', 'timestamp': '2026-01-01', 'event_id': str(index)}
                   for index in range(128)]
    smoke_rows = [{'question_id': 'synthetic', 'effective_timestamp': '2026-01-02',
                   'visible_atoms': 128, 'split': 'development', 'status': 'ok',
                   'feature_cache': {'evictions': 0, 'misses': 128}}]
    smoke_traffic = emit_shared_traffic(smoke_atoms, smoke_rows, dimension, smoke / 'input.trace')
    driver = (args.output / 'native/driver').resolve()
    smoke_result = run_native(smoke.resolve(), driver, smoke_traffic)
    (smoke / 'qualification.json').write_text(json.dumps(smoke_result, indent=2))
    if not smoke_result['host_trace_qualified']:
        raise ValueError('native synthetic smoke failed; real trace was not executed')
    report = run_native(args.output.resolve(), driver, traffic)
    frozen_sources.update(frozen_inputs)
    mismatches = [path for path, expected in frozen_sources.items() if digest(path) != expected]
    report.update({'complete': True, 'source_question_ids': result['selected_question_ids'],
                   'frozen_source_mismatches': mismatches, 'source_result_sha256': digest(args.result),
                   'driver_sha256': digest(driver), 'trace_sha256': digest(args.output / 'input.trace'),
                   'traffic_contract': traffic['traffic_contract'],
                   'synthetic_smoke_qualified': smoke_result['host_trace_qualified']})
    report['host_trace_qualified'] &= not mismatches
    (args.output / 'qualification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['host_trace_qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
