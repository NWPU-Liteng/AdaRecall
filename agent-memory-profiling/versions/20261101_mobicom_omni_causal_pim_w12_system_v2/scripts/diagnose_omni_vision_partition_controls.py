"""Frozen first-multi-image controls; diagnostic only, never promotion evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch
from PIL import Image

from audit_omni_smoke_closure import digest, verify_source_snapshot
from run_omni_vision_partition_probe import compare_features


def stage_comparisons(reference, candidate):
    if list(reference) != list(candidate):
        raise ValueError('identical ordered stage identities required')
    return [{'stage': name, **compare_features(reference[name], candidate[name])}
            for name in reference]


def capture_stages(visual, token_count, pooled_count):
    captured, hooks = {}, []
    stages = [('patch_embed', visual.patch_embed)]
    stages.extend((f'block_{index}', block) for index, block in enumerate(visual.blocks))
    stages.append(('merger', visual.merger))
    for name, module in stages:
        count = pooled_count if name == 'merger' else token_count

        def capture_output(unused_module, unused_inputs, output, stage=name, length=count):
            if not isinstance(output, torch.Tensor) or output.shape[0] < length:
                raise ValueError('unexpected vision stage output')
            captured[stage] = output[:length].detach().to(device='cpu', copy=True).contiguous()

        hooks.append(module.register_forward_hook(capture_output))
    return captured, hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text())
    registration = json.loads(args.smoke.with_suffix('.registration.json').read_text())
    if (not smoke.get('complete') or not smoke.get('verify_generation')
            or smoke.get('evaluation_label_files_read') != []
            or not verify_source_snapshot(smoke, registration)
            or registration['question_ids'] != [row['question_id'] for row in smoke['rows']]):
        raise ValueError('frozen independently verified smoke required')
    request = next(row for row in smoke['rows'] if len(row['images']) > 1)
    paths = [Path(__file__), Path(__file__).with_name('run_omni_vision_partition_probe.py'),
             Path(__file__).with_name('omni_visual_feature_store.py'),
             Path(__file__).with_name('audit_omni_smoke_closure.py')]
    sources = {str(path): digest(path) for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'reserved-before-model-execution'}, handle)
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(exist_ok=False)
    for path in paths:
        target = snapshot / path.name
        target.write_bytes(path.read_bytes())
        if digest(target) != sources[str(path)]:
            raise ValueError('source snapshot differs')
    with args.output.with_suffix('.registration.json').open('x') as handle:
        json.dump({'question_id': request['question_id'], 'producer_smoke_sha256': digest(args.smoke),
            'source_sha256': sources, 'selection': 'first multi-image request in frozen verified order',
            'controls': ['batch-repeat', 'single-repeat', 'first-image-stage-localization'],
            'evaluation_label_files_read': []}, handle, indent=2)
    if digest(args.image_archive) != smoke['archive_sha256']:
        raise ValueError('image archive identity differs')
    reader_path = Path(smoke['reader'])
    for name, expected in smoke['visual_state_identity_files_sha256'].items():
        if Path(name).name != name or digest(reader_path / name) != expected:
            raise ValueError('checkpoint identity differs')
    checkpoint_stats = {name: ((reader_path / name).stat().st_size,
                               (reader_path / name).stat().st_mtime_ns)
                        for name in smoke['visual_state_identity_files_sha256']}
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    import transformers.vision_utils as vision_utils
    environment_paths = [Path(modeling.__file__), Path(vision_utils.__file__)]
    environment_hashes = {str(path): digest(path) for path in environment_paths}
    processor = AutoProcessor.from_pretrained(str(reader_path), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    reader = AutoModelForImageTextToText.from_pretrained(str(reader_path), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    images = []
    with zipfile.ZipFile(args.image_archive) as archive:
        for image in request['images']:
            content = archive.read(image['archive_member'])
            if hashlib.sha256(content).hexdigest() != image['sha256']:
                raise ValueError('original image differs')
            images.append(Image.open(io.BytesIO(content)).convert('RGB'))
    inputs = processor.image_processor(images=images, return_tensors='pt').to(args.device)
    pixels, grids = inputs['pixel_values'], inputs['image_grid_thw']
    sizes = grids.prod(-1).tolist()
    first_size = sizes[0]
    pooled_size = first_size // reader.model.visual.spatial_merge_size**2
    with torch.inference_mode():
        batch_stages, hooks = capture_stages(reader.model.visual, first_size, pooled_size)
        try:
            batch = reader.model.get_image_features(pixels, grids, return_dict=True, use_cache=True)
        finally:
            for hook in hooks:
                hook.remove()
        batch_repeat = reader.model.get_image_features(pixels, grids, return_dict=True, use_cache=True)
        batch_repeat_checks = [compare_features(first, second) for first, second in zip(
            batch.pooler_output, batch_repeat.pooler_output)]
        single_checks, partition_checks, first_single_stages = [], [], None
        for index, (chunk, grid) in enumerate(zip(pixels.split(sizes), grids)):
            single_stages, hooks = capture_stages(reader.model.visual, first_size, pooled_size) if index == 0 else ({}, [])
            try:
                single = reader.model.get_image_features(chunk, grid.unsqueeze(0), return_dict=True, use_cache=True)
            finally:
                for hook in hooks:
                    hook.remove()
            repeated = reader.model.get_image_features(chunk, grid.unsqueeze(0), return_dict=True, use_cache=True)
            single_checks.append(compare_features(single.pooler_output[0], repeated.pooler_output[0]))
            partition_checks.append(compare_features(batch.pooler_output[index], single.pooler_output[0]))
            if index == 0:
                first_single_stages = single_stages
        localized = stage_comparisons(batch_stages, first_single_stages)
    if any(digest(Path(name)) != expected for name, expected in {**sources, **environment_hashes}.items()):
        raise ValueError('execution source changed')
    for name, expected in checkpoint_stats.items():
        stat = (reader_path / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError('checkpoint changed during execution')
    result = {'schema_version': 'omni-vision-partition-controls-v1', 'complete': True,
        'question_id': request['question_id'], 'images': request['images'],
        'batch_repeat': batch_repeat_checks, 'single_repeat': single_checks,
        'batch_vs_partition': partition_checks, 'first_image_stages': localized,
        'first_differing_stage': next((value['stage'] for value in localized if not value['bitwise_equal']), None),
        'source_sha256': sources, 'environment_source_sha256': environment_hashes,
        'producer_smoke_sha256': digest(args.smoke), 'archive_sha256': smoke['archive_sha256'],
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda_device': torch.cuda.get_device_name(torch.device(args.device)),
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'evaluation_label_files_read': [], 'full_generation_equivalence_qualified': False,
        'mainline_promotion_authorized': False, 'end_to_end_speedup': None,
        'boundary': 'single registered multi-image request and first-image layer localization; repeat controls identify empirical stability only; no quality, kernel-causality, general determinism or speedup claim'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({name: result[name] for name in ('complete', 'question_id', 'first_differing_stage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
