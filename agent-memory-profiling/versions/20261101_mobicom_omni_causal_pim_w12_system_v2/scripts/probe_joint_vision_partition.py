"""Frozen-prefix Qwen3-VL visual partition and operator diagnostic."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import torch
from PIL import Image

from omni_evidence_planner import images_for_atoms
from omni_joint_index import TrainedMultimodalReranker, digest, model_files
from omni_joint_vision_contract import compare_partition_outputs
from omni_visual_feature_store import VisualFeatureStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    source = json.loads(args.source_run.read_text())
    if not source.get('complete') or source.get('evaluation_label_files_read') != []:
        raise ValueError('complete label-free source required')
    if source.get('relevance_modality') != 'joint_index' or source.get('screening_run') is not True:
        raise ValueError('joint-index development screening required')
    row = source['rows'][0]
    if row['person_id'] not in {'0', '1', '2'}:
        raise ValueError('development users only')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'reserved_before_execution'}, handle)
    trace = args.output.with_suffix('.stacks.log').open('x')
    faulthandler.dump_traceback_later(60, repeat=True, file=trace)
    script_root = Path(__file__).resolve().parent
    source_names = ['probe_joint_vision_partition.py', 'omni_evidence_planner.py',
                    'omni_joint_index.py', 'omni_service_measurement.py',
                    'omni_visual_feature_store.py', 'run_omni_vision_partition_probe.py',
                    'audit_omni_smoke_closure.py', 'omni_joint_vision_contract.py']
    sources = {name: digest(script_root / name) for name in source_names}
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(exist_ok=False)
    for name in source_names:
        (snapshot / name).write_bytes((script_root / name).read_bytes())
        if digest(snapshot / name) != sources[name]:
            raise ValueError('source changed while snapshotting')
    plan = row['hypergraph_evidence_plan']
    edges = plan['candidate_replay'][:2]
    records = plan['multimodal_relevance_records'][:2]
    if len(edges) != 2 or len(records) != 2:
        raise ValueError('frozen request must contain two candidates and two relevance records')
    registration = {'status': 'registered_before_model_execution', 'sources_sha256': sources,
        'source_run_sha256': digest(args.source_run), 'question_id': row['question_id'],
        'edge_ids': [edge['edge_id'] for edge in edges], 'image_max_pixels': source['image_max_pixels'],
        'selection': 'first two candidates of first frozen development request; no labels',
        'evaluation_label_files_read': [], 'profile_reference_only': True,
        'instrumentation': 'synchronized module boundary timers; no torch profiler',
        'boundary': 'feature/operator gate only; no answer, accuracy, or E2E speedup claim'}
    args.output.with_suffix('.registration.json').write_text(json.dumps(registration, indent=2))
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if digest(args.prepared / 'audit.json') != source['prepared_audit_sha256']:
        raise ValueError('prepared release changed')
    if digest(args.prepared / 'omni_graph.json') != audit['artifacts_sha256']['omni_graph.json']:
        raise ValueError('observable graph changed')
    if digest(args.image_archive) != source['archive_sha256']:
        raise ValueError('archive identity changed')
    identity = model_files(args.model)
    if identity != source['trained_multimodal_reranker_identity']:
        raise ValueError('reranker model changed')
    graph = next(item for item in json.loads((args.prepared / 'omni_graph.json').read_text())
                 if item['person_id'] == row['person_id'])
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    images, checksums = [], []
    with zipfile.ZipFile(args.image_archive) as archive:
        for edge, record in zip(edges, records, strict=True):
            if record['edge_id'] != edge['edge_id']:
                raise ValueError('candidate identity changed')
            loaded_hashes = []
            for ref in images_for_atoms([atoms[identity] for identity in edge['atom_ids']]):
                content = archive.read(ref.get('resolved', ref['normalized']))
                images.append(Image.open(io.BytesIO(content)).convert('RGB'))
                loaded_hashes.append(hashlib.sha256(content).hexdigest())
            if loaded_hashes != record['image_sha256']:
                raise ValueError('image payload changed')
            checksums.extend(loaded_hashes)
    if len(images) < 2:
        raise ValueError('frozen prefix lacks a multi-image partition opportunity')
    reranker = TrainedMultimodalReranker(args.model, args.device, source['image_max_pixels'], batch_size=2)
    model = reranker.wrapper.model
    inputs = reranker.wrapper.processor.image_processor(images=images, return_tensors='pt').to(args.device)
    pixels, grids = inputs['pixel_values'], inputs['image_grid_thw']
    patches = grids.prod(dim=-1).cpu().tolist()
    print(json.dumps({'phase': 'inputs_ready', 'grid': grids.cpu().tolist(),
                      'pixel_shape': list(pixels.shape)}), flush=True)
    sync = lambda: torch.cuda.synchronize(args.device)
    phase_log = args.output.with_suffix('.phases.jsonl').open('x')
    handles, starts, module_times = [], {}, []

    def before(name):
        def hook(module, inputs):
            sync()
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(module, inputs, output):
            sync()
            record = {'module': name, 'seconds': time.perf_counter() - starts[name]}
            module_times.append(record)
            phase_log.write(json.dumps(record) + '\n')
            phase_log.flush()
        return hook

    for name, module in model.visual.named_children():
        targets = [(f'{name}.{index}', child) for index, child in enumerate(module)] if isinstance(
            module, torch.nn.ModuleList) else [(name, module)]
        for label, child in targets:
            handles.append(child.register_forward_pre_hook(before(label)))
            handles.append(child.register_forward_hook(after(label)))
    with torch.inference_mode():
        sync()
        started = time.perf_counter()
        reference_output = model.get_image_features(pixels, grids)
        sync()
        reference_seconds = time.perf_counter() - started
        for handle in handles:
            handle.remove()
        phase_log.close()
        reference_features = reference_output.pooler_output
        reference_deepstack = reference_output.deepstack_features
        partitions = []
        started = time.perf_counter()
        for chunk, grid in zip(pixels.split(patches), grids):
            output = model.get_image_features(chunk, grid.unsqueeze(0))
            partitions.append(output)
        sync()
        partition_seconds = time.perf_counter() - started
        comparisons, intermediate_comparisons = compare_partition_outputs(reference_output, partitions)
    if any(digest(script_root / name) != checksum for name, checksum in sources.items()):
        raise ValueError('probe source changed during execution')
    result = {'complete': True, 'diagnostic_only': True, 'question_id': row['question_id'],
        'source_run_sha256': registration['source_run_sha256'], 'image_sha256': checksums,
        'pixel_values_sha256': VisualFeatureStore.tensor_digest(pixels), 'image_grid_thw': grids.cpu().tolist(),
        'pooled_feature_comparisons': comparisons, 'deepstack_comparisons': intermediate_comparisons,
        'all_features_bitwise_equal': all(item['bitwise_equal'] for item in comparisons + intermediate_comparisons),
        'feature_bytes': sum(value.numel() * value.element_size()
                             for value in list(reference_features) + list(reference_deepstack)),
        'reference_profiled_seconds': reference_seconds, 'partition_seconds_diagnostic_only': partition_seconds,
        'reference_module_times': module_times,
        'evaluation_label_files_read': [], 'mainline_promotion_authorized': False,
        'main_experiment_closed': False, 'actual_pim_end_to_end_speedup': None,
        'boundary': 'first frozen batch only; original run not warmed; synchronized hooks; no throughput or E2E comparison'}
    args.output.write_text(json.dumps(result, indent=2))
    faulthandler.cancel_dump_traceback_later()
    trace.close()
    print(json.dumps({'complete': True, 'all_features_bitwise_equal': result['all_features_bitwise_equal'],
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
