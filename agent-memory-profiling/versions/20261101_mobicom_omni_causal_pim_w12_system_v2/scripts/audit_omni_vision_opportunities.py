"""Reconstruct frozen processor operands and bound natural vision reuse."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import time
import zipfile

import torch
from PIL import Image

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_latency_accounting import full_service_summary
from omni_visual_feature_store import VisualFeatureStore
from run_omni_raw_image_smoke import SYSTEM


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def reconstruct_inputs(processor, archive, row, manifest):
    if (manifest['question_id'] != row['question_id']
            or manifest['person_id'] != row['person_id']
            or manifest['visibility_protocol'] != 'history_end'
            or manifest['effective_timestamp'] != row['effective_timestamp']
            or hashlib.sha256(manifest['question'].encode()).hexdigest() != row['question_sha256']):
        raise ValueError('frozen observable manifest differs')
    content = [{'type': 'text', 'text': '原始对话记忆：\n局部对话超边：\n'
        + row['hypergraph_evidence_plan']['text']
        + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
        + f"\n问题：{manifest['question']}\n请依据文字和随后附上的原始图片作答。"}]
    images = []
    for record in row['images']:
        raw = archive.read(record['archive_member'])
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('original image content differs')
        decoded = Image.open(io.BytesIO(raw)).convert('RGB')
        images.append(decoded)
        content.extend([{'type': 'text', 'text':
            f"以下原图来自会话 {record['source_session_id']} 的对话轮 {record['source_atom_id']}："},
            {'type': 'image', 'image': decoded}])
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=images, return_tensors='pt')
    hashes = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
              if isinstance(value, torch.Tensor)}
    if (hashes != row['processor_inputs_sha256']
            or inputs.input_ids.shape[1] != row['input_tokens']
            or inputs.pixel_values.shape[0] != row['visual_patches']):
        raise ValueError('actual full processor tensors differ, not just shape')
    return inputs


def image_operands(inputs, images, execution_identity, merge_size, output_width):
    grids, pixels = inputs['image_grid_thw'], inputs['pixel_values']
    if (merge_size < 1 or output_width < 1 or grids.shape != (len(images), 3)
            or grids.dtype not in (torch.int32, torch.int64)
            or pixels.ndim != 2 or not pixels.is_floating_point()
            or torch.any(grids <= 0)):
        raise ValueError('positive image grids and floating patch tensor required')
    sizes = grids.prod(-1).tolist()
    if sum(sizes) != pixels.shape[0] or any(size % merge_size**2 for size in sizes):
        raise ValueError('patch/grid layout mismatch')
    result = []
    for record, chunk, grid, size in zip(images, pixels.split(sizes), grids, sizes):
        operand = {'raw_sha256': record['sha256'],
            'pixel_values_sha256': VisualFeatureStore.tensor_digest(chunk),
            'grid_sha256': VisualFeatureStore.tensor_digest(grid.unsqueeze(0)),
            'grid_thw': grid.tolist(), 'patches': size,
            'pooled_payload_bytes': size // merge_size**2 * output_width * 2,
            'execution_identity': execution_identity}
        result.append({**operand, 'operand_sha256': identity(operand),
            'source_session_id': record['source_session_id'],
            'source_atom_id': record['source_atom_id']})
    return result


def audit_stream(rows, observations, capacity_bytes):
    if not rows or len(rows) != len(observations) or capacity_bytes < 1:
        raise ValueError('full nonempty stream and positive declared capacity required')
    seen_raw, seen_images, seen_batches, question_ids = set(), set(), set(), set()
    entries = OrderedDict()
    resident = peak = evictions = rejected = 0
    raw_repeats = image_repeats = batch_repeats = batch_hits = 0
    full_bound = partial_bound = total_vision = total_service = 0.
    records = []
    for row, observation in zip(rows, observations):
        question_id = row['question_id']
        if question_id != observation['question_id'] or question_id in question_ids:
            raise ValueError('original complete serving order and unique denominator required')
        question_ids.add(question_id)
        operands = observation['images']
        if not operands or len(operands) != len(row['images']):
            raise ValueError('all original image operands required')
        batch = tuple(value['operand_sha256'] for value in operands)
        raw = [value['raw_sha256'] for value in operands]
        repeated = [value in seen_images for value in batch]
        repeated_raw = [value in seen_raw for value in raw]
        raw_repeats += sum(repeated_raw)
        image_repeats += sum(repeated)
        batch_repeats += batch in seen_batches
        hit = batch in entries
        batch_hits += hit
        vision = row['service_stage_ledger']['nested_seconds']['vision_in_prefill_seconds']
        service = row['service_seconds']
        if not (math.isfinite(vision) and math.isfinite(service) and 0 <= vision <= service):
            raise ValueError('valid nested vision/service timers required')
        total_vision += vision
        total_service += service
        full_bound += vision if hit else 0.
        partial_bound += vision if any(repeated) else 0.
        size = sum(value['pooled_payload_bytes'] for value in operands)
        if hit:
            entries.move_to_end(batch)
        elif size > capacity_bytes:
            rejected += 1
        else:
            while entries and resident + size > capacity_bytes:
                _, old_size = entries.popitem(last=False)
                resident -= old_size
                evictions += 1
            entries[batch] = size
            resident += size
            peak = max(peak, resident)
        records.append({**observation, 'previous_image_operands': repeated,
            'previous_raw_images': repeated_raw, 'whole_batch_resident_hit_opportunity': hit,
            'hypothetical_payload_bytes': size, 'resident_payload_bytes': resident,
            'vision_in_prefill_seconds': vision, 'service_seconds': service,
            'embedding_producer_sha256s': [value['producer_sha256'] for value in row['embedding_producers']],
            'accepted_atom_ids': row['hypergraph_evidence_plan']['accepted_atom_ids'],
            'legality': 'operand identity only; independent-image output equivalence not established'})
        seen_batches.add(batch)
        seen_images.update(batch)
        seen_raw.update(raw)
    return {'requests': len(rows), 'images': sum(len(value['images']) for value in observations),
        'unique_raw_images': len(seen_raw), 'unique_image_operands': len(seen_images),
        'previous_request_raw_image_occurrences': raw_repeats,
        'previous_request_image_operand_occurrences': image_repeats,
        'ordered_batch_repeats': batch_repeats, 'bounded_whole_batch_hit_opportunities': batch_hits,
        'capacity_bytes': capacity_bytes, 'peak_payload_bytes': peak, 'resident_payload_bytes': resident,
        'evictions': evictions, 'oversize_rejections': rejected,
        'total_service_seconds': total_service, 'total_vision_seconds': total_vision,
        'zero_cost_whole_batch_time_removal_bound_seconds': full_bound,
        'zero_cost_independent_image_loose_time_removal_bound_seconds': partial_bound,
        'bound_definition': 'whole-batch removes all observed vision time on resident hits; independent-image bound removes all vision time even on partial overlap, deliberately loose; lookup, validation and transfer assumed zero; retrospective server bounds, not measured gains or LPDDR predictions',
        'rows': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-cache-bytes', type=int, default=64 * 1024**2)
    args = parser.parse_args()
    if args.output.exists() or args.visual_cache_bytes < 1:
        raise ValueError('fresh output and positive capacity required')
    plan = json.loads(args.cohort_plan.read_text())
    paths = [args.cohort_plan, Path(__file__), Path(__file__).with_name('run_omni_raw_image_smoke.py'),
             Path(__file__).with_name('omni_visual_feature_store.py')]
    runs, registrations = [], []
    for job in plan['jobs']:
        path = Path(job['output'])
        run = json.loads(path.read_text())
        registration_path = path.with_suffix('.registration.json')
        registration = json.loads(registration_path.read_text())
        if (not run['complete'] or run['evaluation_label_files_read'] != []
                or not verify_source_snapshot(run, registration)
                or [row['question_id'] for row in run['rows']] != job['question_ids']
                or registration['question_ids'] != job['question_ids']
                or any(row['person_id'] != job['person_id'] for row in run['rows'])
                or hashlib.sha256(SYSTEM.encode()).hexdigest() != run['system_prompt_sha256']):
            raise ValueError('complete original label-free user streams required')
        runs.append(run)
        registrations.append(registration)
        paths.extend([path, registration_path])
    full_service_summary(runs)
    common = registrations[0]['arguments']
    archive_path = Path(common['image_archive'])
    prepared = Path(common['prepared'])
    manifest_path = prepared / 'manifests.jsonl'
    paths.extend([archive_path, manifest_path, prepared / 'audit.json'])
    reader_path = Path(runs[0]['reader'])
    config_paths = sorted(path for path in reader_path.iterdir()
                          if path.suffix in {'.json', '.txt', '.jinja', '.safetensors'})
    paths.extend(config_paths)
    hashes = {str(path): digest(path) for path in paths}
    for run, registration in zip(runs, registrations):
        if (run['reader'] != str(reader_path) or run['image_max_pixels'] != runs[0]['image_max_pixels']
                or hashes[str(archive_path)] != run['archive_sha256']
                or hashes[str(prepared / 'audit.json')] != run['prepared_audit_sha256']
                or registration['arguments']['prepared'] != str(prepared)
                or registration['arguments']['image_archive'] != str(archive_path)
                or any(hashes[str(reader_path / name)] != expected
                       for name, expected in run['visual_state_identity_files_sha256'].items())):
            raise ValueError('immutable model, archive or preparation identity differs')
    prepared_audit = json.loads((prepared / 'audit.json').read_text())
    if hashes[str(manifest_path)] != prepared_audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving manifest identity differs')
    stats = {name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns) for name in hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'registered-before-processor-execution'}, handle)
    with args.output.with_suffix('.registration.json').open('x') as handle:
        json.dump({'inputs_sha256': hashes, 'question_ids': [row['question_id'] for run in runs for row in run['rows']],
            'capacity_bytes_per_user': args.visual_cache_bytes,
            'selection': 'all original requests and order; no labels, no Reader generation'}, handle, indent=2)
    from transformers import AutoProcessor
    import transformers
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(reader_path), local_files_only=True)
    processor.image_processor.size['longest_edge'] = runs[0]['image_max_pixels']
    execution_identity = identity({'model': runs[0]['reader_identity'],
        'processor_files': {path.name: hashes[str(path)] for path in config_paths if path.suffix != '.safetensors'},
        'processor': processor.image_processor.to_dict(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'encoder_dtype': 'torch.bfloat16', 'attention': 'sdpa'})
    vision_config = json.loads((reader_path / 'config.json').read_text())['vision_config']
    manifests = {row['question_id']: row for row in map(json.loads, manifest_path.read_text().splitlines())}
    users = []
    with zipfile.ZipFile(archive_path) as archive:
        for run in runs:
            observations = []
            for row in run['rows']:
                inputs = reconstruct_inputs(processor, archive, row, manifests[row['question_id']])
                observations.append({'question_id': row['question_id'],
                    'processor_inputs_sha256': row['processor_inputs_sha256'],
                    'images': image_operands(inputs, row['images'], execution_identity,
                        vision_config['spatial_merge_size'], vision_config['out_hidden_size'])})
            users.append({'person_id': run['rows'][0]['person_id'],
                **audit_stream(run['rows'], observations, args.visual_cache_bytes)})
    for name, expected in stats.items():
        if (Path(name).stat().st_size, Path(name).stat().st_mtime_ns) != expected:
            raise ValueError('input changed during reconstruction')
    totals = {name: sum(user[name] for user in users) for name in (
        'requests', 'images', 'previous_request_raw_image_occurrences', 'previous_request_image_operand_occurrences',
        'ordered_batch_repeats', 'bounded_whole_batch_hit_opportunities', 'total_service_seconds',
        'total_vision_seconds', 'zero_cost_whole_batch_time_removal_bound_seconds',
        'zero_cost_independent_image_loose_time_removal_bound_seconds')}
    for kind in ('whole_batch', 'independent_image_loose'):
        saved = totals[f'zero_cost_{kind}_time_removal_bound_seconds']
        totals[f'zero_cost_{kind}_server_speedup_upper_bound'] = totals['total_service_seconds'] / (totals['total_service_seconds'] - saved)
    result = {'schema_version': 'omni-actual-vision-opportunity-v1', 'complete': True,
        'processor_tensor_equivalence': 'pass', 'inputs_sha256': hashes, 'execution_identity': execution_identity,
        'processor_reconstruction_seconds': time.perf_counter() - started,
        'totals': totals, 'users': users, 'evaluation_label_files_read': [],
        'actual_visual_reuse_executed': False, 'independent_image_output_equivalence_qualified': False,
        'end_to_end_speedup': None, 'pim_end_to_end_speedup': None,
        'boundary': 'past-only natural input identity and payload opportunity audit; no new Reader outputs, no selection by success, no measured saved time; BF16 per-image partition legality remains unqualified'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, **totals}), flush=True)


if __name__ == '__main__':
    main()
