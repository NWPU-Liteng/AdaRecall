"""Frozen-prefix GPU gate for single-image materialization versus original batches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import torch
from PIL import Image

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_visual_feature_store import VisualFeatureStore, synchronize


def compare_features(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {'same_shape_dtype': False, 'bitwise_equal': False, 'max_absolute_difference': None}
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    return {'same_shape_dtype': True, 'finite': finite,
        'bitwise_equal': finite and VisualFeatureStore.tensor_digest(reference) == VisualFeatureStore.tensor_digest(candidate),
        'max_absolute_difference': float((reference.float() - candidate.float()).abs().max().item())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--visual-cache-bytes', type=int, default=64 * 1024**2)
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text())
    registration = json.loads(args.smoke.with_suffix('.registration.json').read_text())
    if (args.limit < 1 or args.limit > len(smoke['rows']) or not smoke.get('complete')
            or not smoke.get('verify_generation') or smoke.get('evaluation_label_files_read') != []
            or not smoke.get('visual_state_identity_files_sha256')
            or not verify_source_snapshot(smoke, registration)
            or [row['question_id'] for row in smoke['rows']] != registration['question_ids']):
        raise ValueError('frozen independently verified development prefix required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'reserved_before_model_execution'}, handle)
    source_paths = [Path(__file__), Path(__file__).with_name('omni_visual_feature_store.py')]
    source_hashes = {str(path): digest(path) for path in source_paths}
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(exist_ok=False)
    for path in source_paths:
        target = snapshot / path.name
        target.write_bytes(path.read_bytes())
        if digest(target) != source_hashes[str(path)]:
            raise ValueError('source changed while snapshotting')
    selected = smoke['rows'][:args.limit]
    registration_result = {'question_ids': [row['question_id'] for row in selected],
        'selection': 'first N requests in pre-existing frozen verified serving order; no labels',
        'producer_smoke_sha256': digest(args.smoke), 'source_sha256': source_hashes,
        'capacity_bytes': args.visual_cache_bytes,
        'scope': 'feature partition legality only; no new answer generation or semantic evaluation'}
    with args.output.with_suffix('.registration.json').open('x') as handle:
        json.dump(registration_result, handle, ensure_ascii=False, indent=2)
    if digest(args.image_archive) != smoke['archive_sha256']:
        raise ValueError('original image archive identity differs')
    reader_path = Path(smoke['reader'])
    checkpoint_hashes = smoke['visual_state_identity_files_sha256']
    for name, expected in checkpoint_hashes.items():
        if Path(name).name != name or digest(reader_path / name) != expected:
            raise ValueError('checkpoint immutable content identity differs')
    checkpoint_stats = {name: ((reader_path / name).stat().st_size, (reader_path / name).stat().st_mtime_ns)
                        for name in checkpoint_hashes}
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers
    processor = AutoProcessor.from_pretrained(str(reader_path), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    processor_identity = hashlib.sha256(json.dumps(processor.image_processor.to_dict(),
                                                  sort_keys=True, default=str).encode()).hexdigest()
    model_identity = hashlib.sha256(json.dumps(checkpoint_hashes, sort_keys=True).encode()).hexdigest()
    reader = AutoModelForImageTextToText.from_pretrained(str(reader_path), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    store = VisualFeatureStore(args.visual_cache_bytes, model_identity, processor_identity,
        output_dtype=reader.model.visual.dtype, spatial_merge_size=reader.model.visual.spatial_merge_size)
    rows = []
    with zipfile.ZipFile(args.image_archive) as archive:
        for request in selected:
            images = []
            for image in request['images']:
                content = archive.read(image['archive_member'])
                if hashlib.sha256(content).hexdigest() != image['sha256']:
                    raise ValueError('pinned original image bytes differ')
                images.append(Image.open(io.BytesIO(content)).convert('RGB'))
            inputs = processor.image_processor(images=images, return_tensors='pt').to(args.device)
            pixels, grids = inputs['pixel_values'], inputs['image_grid_thw']
            before = store.snapshot()
            with torch.inference_mode():
                synchronize(args.device)
                started = time.perf_counter()
                reference = reader.model.get_image_features(pixels, grids, return_dict=True, use_cache=True)
                synchronize(args.device)
                reference_seconds = time.perf_counter() - started
                sizes = grids.prod(-1).tolist()
                candidates = []
                for image, chunk, grid in zip(request['images'], pixels.split(sizes), grids):
                    output = store.resolve_batch(chunk, grid.unsqueeze(0), [image['sha256']],
                        reader.model.get_image_features, return_dict=True, use_cache=True)
                    candidates.append(output.pooler_output[0])
                comparisons = [compare_features(first, second) for first, second in zip(
                    reference.pooler_output, candidates)]
            row = {'question_id': request['question_id'], 'images': request['images'],
                'feature_comparisons': comparisons,
                'all_features_bitwise_equal': all(value['bitwise_equal'] for value in comparisons),
                'reference_batch_encoder_seconds_diagnostic': reference_seconds,
                'cache_before': before, 'cache_after': store.snapshot()}
            rows.append(row)
            args.output.write_text(json.dumps({'complete': False, 'rows': rows}, ensure_ascii=False, indent=2))
            print(json.dumps({'question_id': row['question_id'],
                'all_features_bitwise_equal': row['all_features_bitwise_equal'],
                'single_image_hits': row['cache_after']['hits'] - before['hits']}, ensure_ascii=False), flush=True)
            del reference, candidates, inputs
            torch.cuda.empty_cache()
    if any(digest(Path(path)) != expected for path, expected in source_hashes.items()):
        raise ValueError('probe sources changed during execution')
    for name, expected in checkpoint_stats.items():
        stat = (reader_path / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError('Reader checkpoint changed during execution')
    result = {'schema_version': 'omni-vision-single-image-partition-gate-v1', 'complete': True,
        'rows': rows, 'requests': len(rows),
        'bitwise_feature_pass_requests': sum(row['all_features_bitwise_equal'] for row in rows),
        'feature_partition_gate_passed': all(row['all_features_bitwise_equal'] for row in rows),
        'full_generation_equivalence_qualified': False, 'mainline_promotion_authorized': False,
        'cache': store.snapshot(), 'producer_smoke_sha256': digest(args.smoke),
        'source_sha256': source_hashes, 'reader_checkpoint_sha256': checkpoint_hashes,
        'archive_sha256': smoke['archive_sha256'], 'torch': torch.__version__,
        'transformers': transformers.__version__, 'evaluation_label_files_read': [],
        'end_to_end_speedup': None,
        'boundary': 'natural frozen-prefix feature probe only; single-image encoding changes GPU batch shapes; failure blocks promotion; success still requires full generation equivalence and quality validation; no speedup or KV reuse claim'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({name: result[name] for name in ('complete','requests',
        'bitwise_feature_pass_requests','feature_partition_gate_passed','mainline_promotion_authorized')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
