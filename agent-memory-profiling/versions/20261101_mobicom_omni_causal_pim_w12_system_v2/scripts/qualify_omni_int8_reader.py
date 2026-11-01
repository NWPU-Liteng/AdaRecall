"""Whole-generation qualification of one shared W8A8 integer contract."""
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

from audit_omni_smoke_closure import digest, verify_source_snapshot


def generate(reader, inputs, limit, stop_ids):
    import torch
    reader.model.rope_deltas = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = reader(**inputs, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    tokens = []
    stop_reason = 'max_new_tokens'
    for position in range(limit):
        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens.append(int(token.item()))
        if tokens[-1] in stop_ids:
            stop_reason = 'eos'
            break
        if position + 1 < limit:
            output = reader(input_ids=token, past_key_values=output.past_key_values,
                            use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return {'answer_token_ids': tokens, 'stop_reason': stop_reason,
            'prefill_seconds': prefill_seconds, 'generation_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--primary-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.requests < 1:
        raise ValueError('fresh output and positive natural prefix count required')
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pim_backend.functional_decode import Int8LinearContract, Int8ReaderDispatch, materialize_int8_primary
    from qualify_omni_functional_pim import make_inputs
    from omni_patch_projection import configure_patch_projection
    from omni_visual_feature_store import VisualFeatureStore

    run = json.loads(args.run.read_text())
    registration_path = args.run.with_suffix('.registration.json')
    source_registration = json.loads(registration_path.read_text())
    if (not run.get('complete') or run.get('evaluation_label_files_read') != []
            or run['evidence_policy_name'] != 'local_hyperedge'
            or not verify_source_snapshot(run, source_registration)):
        raise ValueError('complete label-isolated local-hyperedge source required')
    if digest(args.prepared / 'audit.json') != run['prepared_audit_sha256']:
        raise ValueError('prepared release differs')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    manifests_path = args.prepared / 'manifests.jsonl'
    if digest(manifests_path) != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('manifests changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifests_path.read_text().splitlines())}
    selected = run['rows'][:args.requests]
    if len(selected) != args.requests or any(manifests[row['question_id']]['split'] != 'development'
                                           for row in selected):
        raise ValueError('whole authorized development prefix required')
    checkpoint = Path(run['reader'])
    for name, expected in run['visual_state_identity_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('Reader checkpoint changed')
    if digest(args.archive) != run['archive_sha256']:
        raise ValueError('image archive changed')
    sources = [Path(__file__), ROOT / 'pim_backend/functional_decode.py',
               ROOT / 'scripts/qualify_omni_functional_pim.py', ROOT / 'scripts/omni_patch_projection.py']
    processor_files = {path.name: digest(path) for path in checkpoint.iterdir()
                       if path.is_file() and path.suffix in {'.json', '.txt', '.jinja'}}
    registration = {'source_run': str(args.run.resolve()), 'source_sha256': digest(args.run),
                    'source_registration_sha256': digest(registration_path),
                    'source_files_sha256': {str(path): digest(path) for path in sources},
                    'processor_files_sha256': processor_files,
                    'requests': [row['question_id'] for row in selected],
                    'selection': 'first source-order development requests; no labels or outcome selection',
                    'max_new_tokens': run['max_new_tokens'], 'torch': torch.__version__,
                    'transformers': transformers.__version__,
                    'arithmetic': 'per-output-row symmetric W8, per-activation-row dynamic A8, exact INT32 accumulation, FP32 scaling, BF16 outputs',
                    'functional_partition': 'whole output matrix versus two independently accumulated output-row groups; not a bank placement or timing model',
                    'state': 'BF16 source retained for independent control; integer state charged additionally',
                    'evaluation_label_files_read': []}
    integer_reference = None
    if args.primary_reference:
        integer_reference = json.loads(args.primary_reference.read_text())
        if (not integer_reference.get('complete') or not integer_reference.get('all_integer_generations_equal')
                or not integer_reference.get('all_bf16_controls_equal_frozen')
                or integer_reference['registration']['source_sha256'] != digest(args.run)
                or [row['question_id'] for row in integer_reference['rows']] != registration['requests']):
            raise ValueError('complete matching previous integer functional prefix required')
        registration['primary_reference_sha256'] = digest(args.primary_reference)
        registration['state'] = 'replace language Linear BF16 primary with shared INT8 primary; prior functional answers reused, no repeated BF16 control generation'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_root = args.output.with_suffix('.sources')
    snapshot_root.mkdir(exist_ok=False)
    for path in sources:
        target = snapshot_root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if digest(target) != registration['source_files_sha256'][str(path)]:
            raise ValueError('source changed during pre-execution freeze')
    registration['source_snapshot_directory'] = str(snapshot_root)
    args.output.with_suffix('.registration.json').write_text(json.dumps(registration, indent=2))
    args.output.write_text(json.dumps({'complete': False, 'registration': registration, 'rows': []}, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    configure_patch_projection(reader.model.visual, run['reader_patch_projection'])
    primary = None
    if integer_reference:
        before = reader.get_memory_footprint()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        primary = materialize_int8_primary(reader)
        torch.cuda.synchronize()
        primary['conversion_seconds'] = time.perf_counter() - started
        primary['before_model_footprint_bytes'] = before
        primary['after_model_footprint_bytes'] = reader.get_memory_footprint()
        primary['conversion_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = run['image_max_pixels']
    eos = reader.config.text_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    rows = []
    with zipfile.ZipFile(args.archive) as archive, torch.inference_mode():
        for original in selected:
            question_id = original['question_id']
            inputs = make_inputs(processor, original, manifests[question_id], archive)
            signatures = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
                          if isinstance(value, torch.Tensor)}
            if signatures != original['processor_inputs_sha256']:
                raise ValueError('reconstructed whole processor inputs differ')
            inputs = inputs.to('cuda:0')
            result = {'question_id': question_id, 'person_id': original['person_id'],
                      'question_sha256': original['question_sha256'], 'processor_inputs_sha256': signatures}
            if integer_reference:
                earlier = {row['question_id']: row for row in integer_reference['rows']}[question_id]
                for mode, partitions in (('int8_host', 1), ('int8_partitioned', 2)):
                    for module in reader.modules():
                        if isinstance(module, Int8LinearContract):
                            module.output_partitions = partitions
                    result[mode] = generate(reader, inputs, run['max_new_tokens'], stop_ids)
                    answer = processor.tokenizer.decode(result[mode]['answer_token_ids'], skip_special_tokens=True).strip()
                    result[mode].update(answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
                result['bf16_control_matches_frozen'] = earlier['bf16_control_matches_frozen']
                result['bf16_control_reexecuted'] = False
                result['primary_matches_previous_int8_host'] = (
                    result['int8_host']['answer_token_ids'] == earlier['int8_host']['answer_token_ids']
                    and result['int8_host']['stop_reason'] == earlier['int8_host']['stop_reason'])
                result['int8_partitioned_tokens_equal_host'] = (
                    result['int8_host']['answer_token_ids'] == result['int8_partitioned']['answer_token_ids']
                    and result['int8_host']['stop_reason'] == result['int8_partitioned']['stop_reason'])
                rows.append(result)
                args.output.write_text(json.dumps({'complete': False, 'registration': registration,
                    'primary_storage': primary, 'rows': rows}, ensure_ascii=False, indent=2))
                print(json.dumps({'question_id': question_id,
                    'primary_matches_previous_int8': result['primary_matches_previous_int8_host'],
                    'integer_partition_matches': result['int8_partitioned_tokens_equal_host']}), flush=True)
                continue
            control = generate(reader, inputs, run['max_new_tokens'], stop_ids)
            result['bf16_control_matches_frozen'] = (control['answer_token_ids'] == original['answer_token_ids']
                                                    and control['stop_reason'] == original['stop_reason'])
            result['bf16'] = control
            torch.cuda.reset_peak_memory_stats()
            with Int8ReaderDispatch(reader) as dispatcher:
                state_bytes = dispatcher.storage_bytes
                conversion_seconds = dispatcher.quantization_seconds
                module_count = len(dispatcher.layouts)
                for mode, partitions in (('int8_host', 1), ('int8_partitioned', 2)):
                    dispatcher.output_partitions = partitions
                    result[mode] = generate(reader, inputs, run['max_new_tokens'], stop_ids)
                    answer = processor.tokenizer.decode(result[mode]['answer_token_ids'], skip_special_tokens=True).strip()
                    result[mode].update(answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
                result['integer_dispatch_calls'] = dispatcher.calls
            result.update(integer_state_bytes=state_bytes, quantization_seconds=conversion_seconds,
                          converted_modules=module_count,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          int8_partitioned_tokens_equal_host=(result['int8_host']['answer_token_ids'] ==
                              result['int8_partitioned']['answer_token_ids'] and result['int8_host']['stop_reason'] ==
                              result['int8_partitioned']['stop_reason']))
            rows.append(result)
            args.output.write_text(json.dumps({'complete': False, 'registration': registration, 'rows': rows},
                                               ensure_ascii=False, indent=2))
            print(json.dumps({'question_id': question_id, 'bf16_matches': result['bf16_control_matches_frozen'],
                              'integer_partition_matches': result['int8_partitioned_tokens_equal_host'],
                              'integer_state_bytes': state_bytes}), flush=True)
    for path in sources:
        if digest(path) != registration['source_files_sha256'][str(path)]:
            raise ValueError('qualification source changed during generation')
    result = {'complete': True, 'registration': registration, 'rows': rows,
              'primary_storage': primary,
              'all_integer_generations_equal': all(row['int8_partitioned_tokens_equal_host'] for row in rows),
              'all_bf16_controls_equal_frozen': all(row['bf16_control_matches_frozen'] for row in rows),
              'quality_qualified': False, 'full_system_speedup': None,
              'boundary': 'functional development qualification only; primary storage mode explicitly recorded, transient conversion peak charged; prior BF16 controls may be inherited only from the exact source reference; no physical PIM, placement, latency, or semantic-losslessness claim'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({name: value for name, value in result.items() if name not in {'registration', 'rows'}}), flush=True)


if __name__ == '__main__':
    main()
