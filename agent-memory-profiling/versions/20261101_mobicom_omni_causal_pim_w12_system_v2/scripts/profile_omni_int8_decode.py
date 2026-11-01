"""Attribute executed CUDA work on frozen natural decode, never phone/PIM latency."""
from __future__ import annotations

import argparse
import bisect
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_smoke_closure import digest, verify_source_snapshot
from qualify_omni_functional_pim import make_inputs


def event_summary(profiler):
    events = profiler.key_averages()
    scopes = {event.key: {'calls': event.count,
        'inclusive_device_microseconds': event.device_time_total,
        'inclusive_cpu_microseconds': event.cpu_time_total,
        'self_cpu_microseconds': event.self_cpu_time_total}
        for event in events if event.key.startswith('omni.') and event.device_type.name == 'CPU'}
    required = ('omni.decode_step', 'omni.int8.linear.selected', 'omni.int8.linear.host',
                'omni.int8.quantization', 'omni.int8.integer.selected', 'omni.int8.integer.host')
    if any(name not in scopes for name in required):
        raise ValueError('complete selected/Host Linear and quantization profile required')
    device = {name: scopes[name]['inclusive_device_microseconds'] for name in required}
    linears = device['omni.int8.linear.selected'] + device['omni.int8.linear.host']
    products = device['omni.int8.integer.selected'] + device['omni.int8.integer.host']
    quantization = device['omni.int8.quantization']
    total = device['omni.decode_step']
    if min(total, linears, products, quantization) <= 0:
        raise ValueError('CUDA kernel durations required; CPU-only profile cannot substitute')
    partition_valid = linears <= total + 1e-6 and products + quantization <= linears + 1e-6
    return {'scopes': scopes,
        'scope_partition_valid': partition_valid,
        'raw_scope_totals_microseconds': {'total': total, 'linears': linears,
                                        'products': products, 'quantization': quantization},
        'raw_inclusive_activity_ratios': {
            'selected_linear_over_decode': device['omni.int8.linear.selected'] / total,
            'host_linear_over_decode': device['omni.int8.linear.host'] / total,
            'selected_integer_product_over_decode': device['omni.int8.integer.selected'] / total,
            'host_integer_product_over_decode': device['omni.int8.integer.host'] / total,
            'quantization_over_decode': quantization / total},
        'kernel_work_partition_microseconds': {
            'quantization': quantization, 'selected_integer_product': device['omni.int8.integer.selected'],
            'host_integer_product': device['omni.int8.integer.host'],
            'linear_remainder_including_scale_bias_cast_checks_and_concat': linears - products - quantization,
            'nonlinear_attention_state_and_token_decision': total - linears} if partition_valid else None,
        'selected_integer_fraction_of_profiled_device_work': device['omni.int8.integer.selected'] / total if partition_valid else None,
        'all_integer_fraction_of_profiled_device_work': products / total if partition_valid else None,
        'top_cpu_operations_by_self_time': [
            {'name': event.key, 'calls': event.count,
             'self_cpu_microseconds': event.self_cpu_time_total,
             'self_device_microseconds': event.self_device_time_total}
            for event in sorted((event for event in events if not event.key.startswith('omni.')),
                                key=lambda event: event.self_cpu_time_total, reverse=True)[:20]],
        'boundary': 'sum of CUDA activity durations attributed to CPU record scopes, not critical-path wall time; profiler/synchronization overhead visible separately; not target architecture costs'}


def trace_kernel_attribution(trace_path):
    events = json.loads(trace_path.read_text())['traceEvents']
    annotations = {}
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') != 'user_annotation':
            continue
        name = event.get('name', '')
        if not name.startswith('omni.'):
            continue
        key = (event.get('pid'), event.get('tid'), name)
        annotations.setdefault(key, []).append((event['ts'], event['ts'] + event['dur']))
    for values in annotations.values():
        values.sort()
    starts = {key: [item[0] for item in values] for key, values in annotations.items()}
    cpu_by_external_id = {event.get('args', {}).get('External id'): event for event in events
                          if event.get('ph') == 'X' and event.get('cat') == 'cpu_op'
                          and 'External id' in event.get('args', {})}
    runtime_by_correlation = {event.get('args', {}).get('correlation'): event for event in events
                              if event.get('ph') == 'X' and event.get('cat') in {'cuda_runtime', 'cuda_driver'}
                              and 'correlation' in event.get('args', {})}

    def inside(event, name):
        key = (event.get('pid'), event.get('tid'), name)
        intervals = annotations.get(key, [])
        index = bisect.bisect_right(starts.get(key, []), event['ts']) - 1
        return index >= 0 and event['ts'] <= intervals[index][1]

    totals = {'quantization': 0., 'selected_integer_product': 0.,
              'host_integer_product': 0., 'linear_remaining': 0.,
              'nonlinear_and_token': 0., 'unattributed': 0.}
    counts = {name: 0 for name in totals}
    for event in events:
        if event.get('ph') != 'X' or event.get('cat') not in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
            continue
        args = event.get('args', {})
        owner = cpu_by_external_id.get(args.get('External id')) or runtime_by_correlation.get(args.get('correlation'))
        if owner is None or not inside(owner, 'omni.decode_step'):
            category = 'unattributed'
        elif inside(owner, 'omni.int8.quantization'):
            category = 'quantization'
        elif inside(owner, 'omni.int8.integer.selected'):
            category = 'selected_integer_product'
        elif inside(owner, 'omni.int8.integer.host'):
            category = 'host_integer_product'
        elif inside(owner, 'omni.int8.linear.selected') or inside(owner, 'omni.int8.linear.host'):
            category = 'linear_remaining'
        else:
            category = 'nonlinear_and_token'
        totals[category] += event.get('dur', 0.)
        counts[category] += 1
    total = sum(totals.values()) - totals['unattributed']
    return {'kernel_activity_partition_microseconds': totals,
            'kernel_activity_counts': counts,
            'kernel_activity_partition_valid': total > 0 and totals['unattributed'] == 0,
            'kernel_activity_fractions_of_attributed_decode': {
                name: value / total for name, value in totals.items()
                if name != 'unattributed' and total > 0},
            'boundary': 'non-overlapping attribution by innermost CPU record scope to CUDA kernel/memcpy intervals; activity time is not critical-path wall time'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh profiler artifact required')
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pim_backend import functional_decode
    from pim_backend.pimnast_functional import frozen_route
    from omni_patch_projection import configure_patch_projection
    from omni_visual_feature_store import VisualFeatureStore

    run = json.loads(args.run.read_text())
    registration = json.loads(args.run.with_suffix('.registration.json').read_text())
    if (not run.get('complete') or run.get('reader_quantization') != 'w8a8'
            or run.get('reader_functional_backend', 'host_primary') != 'host_primary'
            or run.get('evaluation_label_files_read') != [] or not verify_source_snapshot(run, registration)):
        raise ValueError('complete frozen label-isolated Host INT8 run required')
    original = run['rows'][0]
    user = str(original['person_id'])
    if user not in {'0', '1', '2'}:
        raise ValueError('development-only profiling required')
    if digest(args.archive) != run['archive_sha256']:
        raise ValueError('frozen image archive changed')
    audit_path = args.prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    manifest_path = args.prepared / 'manifests.jsonl'
    if (digest(audit_path) != run['prepared_audit_sha256']
            or digest(manifest_path) != audit['artifacts_sha256']['manifests.jsonl']):
        raise ValueError('prepared data changed')
    manifests = {item['question_id']: item for item in map(json.loads, manifest_path.read_text().splitlines())}
    manifest = manifests[original['question_id']]
    if (manifest['split'] != 'development'
            or hashlib.sha256(manifest['question'].encode()).hexdigest() != original['question_sha256']):
        raise ValueError('development request identity changed')
    checkpoint = Path(run['reader'])
    for name, expected in run['visual_state_identity_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('Reader model/processor identity changed: ' + name)
    if len(original['answer_token_ids']) < 133:
        raise ValueError('fixed late diagnostic window absent; do not select another request')
    candidates, unused_capacity = frozen_route(json.loads(args.projection.read_text()), user)
    selected = {name for name, item in candidates.items() if item['model_route'] == 'pim'}
    sources = [Path(__file__), ROOT / 'pim_backend/functional_decode.py',
               ROOT / 'scripts/qualify_omni_functional_pim.py', ROOT / 'scripts/omni_patch_projection.py',
               ROOT / 'scripts/omni_visual_feature_store.py', ROOT / 'scripts/run_omni_raw_image_smoke.py',
               ROOT / 'pim_backend/pimnast_functional.py']
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sources:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(path.relative_to(ROOT))] = digest(target)
    frozen_registration = {'run': str(args.run), 'run_sha256': digest(args.run),
        'projection': str(args.projection), 'projection_sha256': digest(args.projection),
        'question_id': original['question_id'], 'person_id': user,
        'selection': 'first request in registered source-order stream; fixed decode windows 1-4 and 129-132',
        'source_files_sha256': hashes, 'source_snapshot': str(snapshot),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'reader_model_processor_identity': run['visual_state_identity_files_sha256'],
        'evaluation_label_files_read': [], 'performance_comparable': False,
        'physical_pim_execution': False}
    args.output.with_suffix('.registration.json').write_text(json.dumps(frozen_registration, indent=2))
    result = {'complete': False, 'registration': frozen_registration, 'windows': [],
              'full_system_speedup': None, 'full_generation_quality': None,
              'timing_origin': 'instrumented_server_cpu_and_cuda_activities'}
    args.output.write_text(json.dumps(result, indent=2))
    torch.manual_seed(0)
    torch.set_num_threads(2)
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    configure_patch_projection(reader.model.visual, run['reader_patch_projection'])
    with torch.inference_mode():
        functional_decode.materialize_int8_primary(reader)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = run['image_max_pixels']
    with zipfile.ZipFile(args.archive) as archive:
        inputs = make_inputs(processor, original, manifest, archive)
    signatures = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
                  if isinstance(value, torch.Tensor)}
    if signatures != original['processor_inputs_sha256']:
        raise ValueError('complete processor input differs from frozen service')
    active = False
    original_quantize = functional_decode.quantize_int8_rows

    def quantize(value):
        with torch.profiler.record_function('omni.int8.quantization') if active else nullcontext():
            return original_quantize(value)

    restores = []
    for name, module in reader.named_modules():
        if not isinstance(module, functional_decode.Int8LinearContract):
            continue
        route = 'selected' if name in selected else 'host'
        old_forward, old_product = module.forward, module.integer_product

        def forward(value, function=old_forward, group=route):
            with torch.profiler.record_function('omni.int8.linear.' + group) if active else nullcontext():
                return function(value)

        def product(value, function=old_product, group=route):
            with torch.profiler.record_function('omni.int8.integer.' + group) if active else nullcontext():
                return function(value)

        restores.append((module, old_forward, old_product))
        module.forward, module.integer_product = forward, product
    functional_decode.quantize_int8_rows = quantize
    profiler = None
    try:
        with torch.inference_mode():
            output = reader(**inputs.to('cuda:0'), use_cache=True, logits_to_keep=1)
            token = output.logits[:, -1, :].argmax(-1, keepdim=True)
            if int(token.item()) != original['answer_token_ids'][0]:
                raise ValueError('prefill token differs from frozen Host')
            started = None
            for step in range(1, 133):
                if step in (1, 129):
                    torch.cuda.synchronize()
                    profiler = torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    profiler.start()
                    active = True
                    started = time.perf_counter()
                with torch.profiler.record_function('omni.decode_step') if active else nullcontext():
                    output = reader(input_ids=token, past_key_values=output.past_key_values,
                                    use_cache=True, logits_to_keep=1)
                    token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                    if int(token.item()) != original['answer_token_ids'][step]:
                        raise ValueError(f'fixed natural token trajectory changed at step {step}')
                if step in (4, 132):
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    active = False
                    profiler.stop()
                    stopped_profiler, profiler = profiler, None
                    trace_path = args.output.with_suffix(f'.steps_{step - 3}_{step}.trace.json')
                    stopped_profiler.export_chrome_trace(str(trace_path))
                    window = {'first_decode_step': step - 3, 'last_decode_step': step,
                              'instrumented_window_wall_seconds': elapsed, **event_summary(stopped_profiler)}
                    window.update(trace_kernel_attribution(trace_path))
                    result['windows'].append(window)
                    window.update(trace_path=str(trace_path), trace_sha256=digest(trace_path))
                    args.output.write_text(json.dumps(result, indent=2))
    finally:
        active = False
        if profiler is not None:
            profiler.stop()
        functional_decode.quantize_int8_rows = original_quantize
        for module, old_forward, old_product in restores:
            module.forward, module.integer_product = old_forward, old_product
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('profiling source changed during replay')
    result.update(complete=True, prefix_tokens_equal_frozen=133, processor_inputs_equal_frozen=True,
        boundary='diagnostic work attribution on two four-step windows of one frozen request; no new full-generation quality, natural reuse opportunity, uninstrumented latency, phone or PIM speedup claim')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'complete': True, 'question_id': original['question_id'],
                     'prefix_tokens_equal_frozen': 133, 'windows': len(result['windows'])}))


if __name__ == '__main__':
    main()
