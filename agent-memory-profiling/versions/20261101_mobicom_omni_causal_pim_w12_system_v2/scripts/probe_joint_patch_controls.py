"""Frozen-prefix patch projection implementation controls, never promotion."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import zipfile

from PIL import Image
import torch
import torch.nn.functional as functional

from omni_evidence_planner import images_for_atoms
from omni_joint_index import TrainedMultimodalReranker, digest, model_files
from run_omni_vision_partition_probe import compare_features


def patch_linear(pixels, projection):
    if (projection.groups != 1 or tuple(projection.padding) != (0, 0, 0)
            or tuple(projection.dilation) != (1, 1, 1)
            or tuple(projection.stride) != tuple(projection.kernel_size)):
        raise ValueError('nonoverlapping full-patch Conv3d required')
    return functional.linear(pixels.to(dtype=projection.weight.dtype),
                             projection.weight.flatten(1), projection.bias)


def timed_call(function, device):
    torch.cuda.synchronize(device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    value = function()
    end.record()
    torch.cuda.synchronize(device)
    return value, {'host_seconds': time.perf_counter() - started,
                   'cuda_event_seconds': begin.elapsed_time(end) / 1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--partition-result', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cudnn-benchmark', action='store_true')
    args = parser.parse_args()
    source = json.loads(args.source_run.read_text())
    gate = json.loads(args.partition_result.read_text())
    if (not source.get('complete') or source.get('evaluation_label_files_read') != []
            or not gate.get('complete') or gate.get('source_run_sha256') != digest(args.source_run)):
        raise ValueError('complete frozen label-free prefix required')
    row = source['rows'][0]
    if row['person_id'] not in {'0', '1', '2'} or row['question_id'] != gate['question_id']:
        raise ValueError('development source identity differs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'reserved_before_execution'}, handle)
    script_root = Path(__file__).resolve().parent
    names = [Path(__file__).name, 'omni_joint_index.py', 'omni_evidence_planner.py',
             'omni_service_measurement.py', 'omni_visual_feature_store.py',
             'run_omni_vision_partition_probe.py', 'audit_omni_smoke_closure.py',
             'omni_joint_vision_contract.py']
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(exist_ok=False)
    sources = {}
    for name in names:
        content = (script_root / name).read_bytes()
        (snapshot / name).write_bytes(content)
        sources[name] = hashlib.sha256(content).hexdigest()
    order = ['conv', 'linear', 'linear', 'conv', 'conv', 'linear']
    registration = {'question_id': row['question_id'], 'source_run_sha256': digest(args.source_run),
        'partition_result_sha256': digest(args.partition_result), 'sources_sha256': sources,
        'order': order, 'warmup_calls_per_implementation': 1,
        'selection': 'same first two frozen candidate images as partition gate',
        'cudnn_benchmark': args.cudnn_benchmark,
        'evaluation_label_files_read': [], 'mainline_promotion_authorized': False,
        'boundary': 'implementation diagnosis, not algorithm novelty or PIM/E2E evidence'}
    with args.output.with_suffix('.registration.json').open('x') as handle:
        json.dump(registration, handle, indent=2)
    if digest(args.prepared / 'audit.json') != source['prepared_audit_sha256']:
        raise ValueError('prepared release changed')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if digest(args.prepared / 'omni_graph.json') != audit['artifacts_sha256']['omni_graph.json']:
        raise ValueError('observable graph changed')
    if digest(args.image_archive) != source['archive_sha256']:
        raise ValueError('image archive changed')
    if model_files(args.model) != source['trained_multimodal_reranker_identity']:
        raise ValueError('model identity changed')
    graph = next(item for item in json.loads((args.prepared / 'omni_graph.json').read_text())
                 if item['person_id'] == row['person_id'])
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    images, image_hashes = [], []
    with zipfile.ZipFile(args.image_archive) as archive:
        for edge in row['hypergraph_evidence_plan']['candidate_replay'][:2]:
            for ref in images_for_atoms([atoms[identity] for identity in edge['atom_ids']]):
                content = archive.read(ref.get('resolved', ref['normalized']))
                image_hashes.append(hashlib.sha256(content).hexdigest())
                images.append(Image.open(io.BytesIO(content)).convert('RGB'))
    if image_hashes != gate['image_sha256']:
        raise ValueError('frozen image identity differs')
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    reranker = TrainedMultimodalReranker(args.model, args.device, source['image_max_pixels'], batch_size=2)
    inputs = reranker.wrapper.processor.image_processor(images=images, return_tensors='pt').to(args.device)
    pixels, grids = inputs['pixel_values'], inputs['image_grid_thw']
    from omni_visual_feature_store import VisualFeatureStore
    if VisualFeatureStore.tensor_digest(pixels) != gate['pixel_values_sha256']:
        raise ValueError('processor tensor differs')
    patch = reranker.wrapper.model.visual.patch_embed
    if patch.proj.weight.device != pixels.device or pixels.device != torch.device(args.device):
        raise ValueError('GPU operand device mismatch')
    operations = {'conv': lambda: patch(pixels), 'linear': lambda: patch_linear(pixels, patch.proj)}
    records, outputs = [], {}
    with args.output.with_suffix('.measurements.jsonl').open('x') as log, torch.inference_mode():
        for name in ('conv', 'linear'):
            value, measurement = timed_call(operations[name], args.device)
            record = {'implementation': name, 'warmup': True, **measurement}
            log.write(json.dumps(record) + '\n')
            log.flush()
            outputs[name] = value
        comparisons = {'conv_vs_linear': compare_features(outputs['conv'], outputs['linear'])}
        reference_vision, reference_vision_time = timed_call(
            lambda: reranker.wrapper.model.get_image_features(pixels, grids), args.device)
        original_projection_forward = patch.proj.forward

        def linearized_projection(hidden_states):
            return patch_linear(hidden_states.reshape(-1, patch.proj.weight[0].numel()), patch.proj)

        patch.proj.forward = linearized_projection
        try:
            linear_vision, linear_vision_time = timed_call(
                lambda: reranker.wrapper.model.get_image_features(pixels, grids), args.device)
        finally:
            patch.proj.forward = original_projection_forward
        vision_comparisons = {
            'pooled': [compare_features(first, second) for first, second in zip(
                reference_vision.pooler_output, linear_vision.pooler_output, strict=True)],
            'deepstack': [compare_features(first, second) for first, second in zip(
                reference_vision.deepstack_features, linear_vision.deepstack_features, strict=True)],
        }
        for name in order:
            value, measurement = timed_call(operations[name], args.device)
            record = {'implementation': name, 'warmup': False, **measurement,
                      'repeat_vs_first': compare_features(outputs[name], value)}
            records.append(record)
            log.write(json.dumps(record) + '\n')
            log.flush()
    medians = {name: statistics.median(item['host_seconds'] for item in records
                                     if item['implementation'] == name) for name in operations}
    if any(digest(script_root / name) != expected for name, expected in sources.items()):
        raise ValueError('probe sources changed during execution')
    result = {'complete': True, 'question_id': row['question_id'], 'source_run_sha256': digest(args.source_run),
        'image_sha256': image_hashes, 'image_grid_thw': grids.cpu().tolist(),
        'pixel_shape': list(pixels.shape), 'input_device': str(pixels.device),
        'weight_device': str(patch.proj.weight.device), 'weight_dtype': str(patch.proj.weight.dtype),
        'weight_shape': list(patch.proj.weight.shape), 'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'comparisons': comparisons, 'measurements': records, 'host_seconds_median': medians,
        'full_vision_original': reference_vision_time,
        'full_vision_linear_patch': linear_vision_time,
        'full_vision_speed_ratio_diagnostic_only': reference_vision_time['cuda_event_seconds'] /
            linear_vision_time['cuda_event_seconds'],
        'full_vision_feature_comparisons': vision_comparisons,
        'evaluation_label_files_read': [], 'mainline_promotion_authorized': False,
        'main_experiment_closed': False, 'actual_pim_end_to_end_speedup': None,
        'boundary': 'patch implementation only; full visual quality and E2E not qualified'}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'complete': True, 'host_seconds_median': medians, 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
