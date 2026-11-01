"""Full-output integer checks on natural Reader decode, without claiming latency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_omni_smoke_closure import digest, verify_source_snapshot


def tensor_digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-generation', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pim_backend.functional_decode import Int8LinearContract, materialize_int8_primary, quantize_int8_rows
    from pim_backend.pimnast_functional import PimnastInt8Layout, PimnastReaderDispatch
    from qualify_omni_functional_pim import make_inputs
    from omni_patch_projection import configure_patch_projection
    from omni_visual_feature_store import VisualFeatureStore
    from qualify_omni_int8_reader import generate

    run = json.loads(args.run.read_text())
    source_registration = json.loads(args.run.with_suffix('.registration.json').read_text())
    if (not run.get('complete') or run.get('evaluation_label_files_read') != []
            or not verify_source_snapshot(run, source_registration)
            or run.get('reader_quantization') != 'w8a8'):
        raise ValueError('complete frozen label-isolated W8A8 source required')
    if (digest(args.prepared / 'audit.json') != run['prepared_audit_sha256']
            or digest(args.archive) != run['archive_sha256']):
        raise ValueError('prepared data or image archive changed')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    manifests_path = args.prepared / 'manifests.jsonl'
    if digest(manifests_path) != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('source manifests changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifests_path.read_text().splitlines())}
    original = run['rows'][0]
    manifest = manifests[original['question_id']]
    if manifest['split'] != 'development' or str(manifest['person_id']) not in {'0', '1', '2'}:
        raise ValueError('only the first authorized development request is permitted')
    checkpoint = Path(run['reader'])
    for name, expected in run['visual_state_identity_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('frozen Reader checkpoint changed: ' + name)
    projection = json.loads(args.projection.read_text())
    profile = next(item for item in projection['per_user'] if item['run_sha256'] == digest(args.run))
    costs = {item['name']: item for item in profile['matrices']}
    selected = {'model.language_model.layers.0.mlp.gate_proj',
                'model.language_model.layers.0.mlp.down_proj',
                'model.language_model.layers.0.linear_attn.in_proj_qkv',
                'model.language_model.layers.11.self_attn.q_proj', 'lm_head'}
    if not selected <= costs.keys():
        raise ValueError('fixed probe projections missing from source execution graph')
    sources = [Path(__file__), ROOT / 'pim_backend/pimnast_functional.py',
               ROOT / 'pim_backend/functional_decode.py', ROOT / 'scripts/qualify_omni_functional_pim.py',
               ROOT / 'scripts/omni_patch_projection.py', ROOT / 'scripts/qualify_omni_int8_reader.py']
    snapshot = args.output.with_suffix('.sources')
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[str(relative)] = digest(target)
        if digest(path) != hashes[str(relative)]:
            raise ValueError('source changed while freezing')
    registration = {'source_run': str(args.run), 'source_run_sha256': digest(args.run),
                    'projection': str(args.projection), 'projection_sha256': digest(args.projection),
                    'question_id': original['question_id'], 'selection': 'first source-order development request',
                    'fixed_projections': sorted(selected), 'source_files_sha256': hashes,
                    'source_snapshot': str(snapshot), 'torch': torch.__version__,
                    'transformers': transformers.__version__,
                    'reader_identity_files_sha256': run['visual_state_identity_files_sha256'],
                    'processor_files_sha256': {path.name: digest(path) for path in checkpoint.iterdir()
                        if path.is_file() and path.suffix in {'.json', '.txt', '.jinja'}},
                    'evaluation_label_files_read': [],
                    'full_generation_requested': args.full_generation,
                    'scope': 'one actual decode vector, full outputs of five fixed projections; not generated-answer quality'}
    args.output.with_suffix('.registration.json').write_text(json.dumps(registration, indent=2))
    result = {'complete': False, 'registration': registration, 'results': [],
              'full_generation_quality_qualified': False, 'full_system_speedup': None,
              'physical_pim_execution': False}
    args.output.write_text(json.dumps(result, indent=2))
    torch.manual_seed(0)
    torch.set_num_threads(2)
    reader = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:0').eval()
    configure_patch_projection(reader.model.visual, run['reader_patch_projection'])
    result['primary_storage'] = materialize_int8_primary(reader)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = run['image_max_pixels']
    captures = {}
    modules = dict(reader.named_modules())
    decode = False

    def make_hook(name):
        def hook(module, inputs, output):
            if decode and name not in captures:
                if not inputs or inputs[0].numel() != module.in_features:
                    raise ValueError('one decode vector required')
                captures[name] = inputs[0].detach().reshape(1, -1).clone()
        return hook

    hooks = [modules[name].register_forward_hook(make_hook(name)) for name in sorted(selected)]
    try:
        with zipfile.ZipFile(args.archive) as archive, torch.inference_mode():
            inputs = make_inputs(processor, original, manifest, archive)
            signatures = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
                          if isinstance(value, torch.Tensor)}
            if signatures != original['processor_inputs_sha256']:
                raise ValueError('reconstructed whole processor inputs differ')
            inputs = inputs.to('cuda:0')
            initial = reader(**inputs, use_cache=True, logits_to_keep=1)
            token = initial.logits[:, -1, :].argmax(-1, keepdim=True)
            decode = True
            reader(input_ids=token, past_key_values=initial.past_key_values,
                   use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            result['processor_inputs_sha256'] = signatures
            result['first_generated_token_matches_frozen'] = int(token.item()) == original['answer_token_ids'][0]
    finally:
        for hook in hooks:
            hook.remove()
    if captures.keys() != selected:
        raise ValueError('some fixed decode projections did not execute')
    raw_root = args.output.with_suffix('.tensors')
    raw_root.mkdir(exist_ok=False)
    with torch.inference_mode():
        for index, name in enumerate(sorted(selected)):
            module = modules[name]
            if not isinstance(module, Int8LinearContract):
                raise ValueError('shared INT8 primary required')
            activation = captures[name]
            quantized, scales = quantize_int8_rows(activation)
            weights = module.weight.detach().cpu().numpy()
            vector = quantized[0].cpu().numpy()
            layout = PimnastInt8Layout(weights, allow_row_padding=True)
            inventory = layout.inventory()
            if inventory['resident_layout_bytes'] != costs[name]['candidate_resident_bytes']:
                raise ValueError('functional bank residency differs from pinned timing model: ' + name)
            started = time.perf_counter()
            actual = layout.gemv(vector)
            elapsed = time.perf_counter() - started
            reference = np.empty(module.out_features, dtype=np.int64)
            for begin in range(0, module.out_features, 1024):
                reference[begin:begin + 1024] = weights[begin:begin + 1024].astype(np.int64) @ vector.astype(np.int64)
            host = module.integer_product(quantized)[0].cpu().numpy()
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(actual, host)
            raw_path = raw_root / f'projection_{index}.npz'
            np.savez(raw_path, activation_bf16_as_fp32=activation.float().cpu().numpy(),
                     quantized_activation=vector, activation_scale=scales.cpu().numpy(),
                     weight_scales=module.scales.cpu().numpy(),
                     pim_int32=actual, host_int32=host, independent_int64=reference)
            row = {'module': name, 'shape': list(weights.shape),
                   'weight_int8_sha256': tensor_digest(weights),
                   'activation_int8_sha256': tensor_digest(vector),
                   'checked_output_elements': int(len(actual)),
                   'integer_outputs_equal_to_int64_and_cuda_primary': True,
                   'max_abs_integer_difference': 0, 'layout': inventory,
                   'projection_route': costs[name]['model_route'],
                   'raw_tensors': str(raw_path), 'raw_tensors_sha256': digest(raw_path),
                   'cpu_functional_emulation_seconds': elapsed,
                   'emulation_seconds_is_not_hardware_latency': True}
            result['results'].append(row)
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps({'module': name, 'integer_outputs_checked': len(actual), 'equal': True}), flush=True)
            del layout, weights
    result['complete'] = True
    if args.full_generation:
        result['complete'] = False
        args.output.write_text(json.dumps(result, indent=2))
        stops = reader.config.text_config.eos_token_id
        stop_ids = set(stops if isinstance(stops, list) else [stops])
        with torch.inference_mode():
            host = generate(reader, inputs, run['max_new_tokens'], stop_ids)
            result['host_generation'] = host
            result['host_full_tokens_match_frozen'] = (
                host['answer_token_ids'] == original['answer_token_ids']
                and host['stop_reason'] == original['stop_reason'])
            args.output.write_text(json.dumps(result, indent=2))
            print(json.dumps({'host_generated_tokens': len(host['answer_token_ids']),
                'host_matches_frozen': result['host_full_tokens_match_frozen']}), flush=True)
            with PimnastReaderDispatch(reader, costs) as dispatcher:
                if sum(layout.storage.numel() for layout in dispatcher.layouts.values()) > profile['declared_pim_pool_bytes']:
                    raise ValueError('PIM padded weights exceed declared pool')
                pim = generate(reader, inputs, run['max_new_tokens'], stop_ids)
                result['pim_functional_generation'] = pim
                result['pim_functional_dispatch'] = dispatcher.summary()
            result['complete_greedy_tokens_and_stop_equal_host'] = (
                host['answer_token_ids'] == pim['answer_token_ids']
                and host['stop_reason'] == pim['stop_reason'])
            result['full_generation_functional_check_pass'] = (
                result['host_full_tokens_match_frozen']
                and result['complete_greedy_tokens_and_stop_equal_host']
                and not result['pim_functional_dispatch']['unobserved_selected_modules'])
            result['quality_scope'] = 'one frozen development request; exact Host/PIM token equality inherits that answer only, not cohort noninferiority'
            result['complete'] = True
            print(json.dumps({'full_generation_functional_check_pass': result['full_generation_functional_check_pass'],
                'pim_tokens': len(pim['answer_token_ids'])}), flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
