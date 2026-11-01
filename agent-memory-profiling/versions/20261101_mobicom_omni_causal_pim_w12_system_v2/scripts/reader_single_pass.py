"""Serve submitted answers with optional immutable cross-request memory checkpoints."""
from __future__ import annotations

import time
import hashlib
import json
import torch

from prefix_state import PrefixStateSnapshot
from residual_profile import ReaderOperatorTrace


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def greedy_decode(model, output, tokenizer, limit, device):
    if limit < 1:
        raise ValueError('positive generation limit required')
    config = getattr(model.config, 'text_config', model.config)
    eos = getattr(config, 'eos_token_id', None)
    eos = tokenizer.eos_token_id if eos is None else eos
    stops = set(eos if isinstance(eos, list) else [eos])
    tokens = []
    cache = output.past_key_values
    for position in range(limit):
        next_id = output.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens.append(int(next_id.item()))
        if tokens[-1] in stops:
            return tokenizer.decode(tokens, skip_special_tokens=True), tokens, 'eos'
        if position + 1 < limit:
            output = model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, 'max_new_tokens'


def read_single_answer(model, processor, question, atoms, args, store, contract, system, atom_text):
    started = time.perf_counter()
    snapshot = None
    memory = ''.join(atom_text(atom) for atom in atoms)
    first_event = atoms[0].get('event_id') if atoms and isinstance(atoms[0], dict) else None
    checkpoint_atoms = []
    for atom in atoms:
        if first_event is not None and atom.get('event_id') != first_event:
            break
        checkpoint_atoms.append(atom)
    checkpoint_memory = ''.join(atom_text(atom) for atom in checkpoint_atoms)
    query_tail = f"查询时间：{question['effective_timestamp']}\n问题：{question['question']}"

    def render(tail, evidence=memory):
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': f'原始记忆证据：\n{evidence}\n' + tail}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)

    inputs = processor(text=[render(query_tail)], return_tensors='pt').to(args.device)
    probe = processor(text=[render('MEMORY_CHECKPOINT_BOUNDARY', checkpoint_memory)], return_tensors='pt')
    first_ids = inputs.input_ids[0].tolist()
    probe_ids = probe.input_ids[0].tolist()
    boundary = 0
    while boundary < min(len(first_ids), len(probe_ids)) and first_ids[boundary] == probe_ids[boundary]:
        boundary += 1
    boundary = min(boundary, len(first_ids) - 1)
    memory_tokens = len(processor.tokenizer.encode(memory, add_special_tokens=False))
    if boundary < 1 or memory_tokens > args.budget_tokens or len(first_ids) > args.budget_tokens + 2048:
        raise ValueError('memory checkpoint or input budget invalid')
    trace = ReaderOperatorTrace(model)
    snapshot_seconds = restore_seconds = checkpoint_prefill_seconds = 0.0
    reused = 0
    validation = {'status': 'not_requested', 'overhead_seconds': 0.0}
    prefix_meta = {'status': 'full_recompute', 'reused_tokens': 0,
                   'checkpoint_scope': 'first_event_block', 'checkpoint_event_id': first_event,
                   'execution_contract': 'unsplit_cold_reference_v1',
                   'prefill_execution': 'unsplit_reference'}
    validate_request = False
    try:
        _sync(args.device)
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            if store is not None:
                snapshot = store.lookup(inputs.input_ids, contract=contract, maximum_tokens=boundary)
            validate_request = args.validate_prefix and snapshot is not None and (
                getattr(args, 'request_index', 0) % getattr(args, 'prefix_audit_every', 1) == 0
                or (snapshot is not None and store.hits == 1))
            if snapshot is not None:
                restore_started = time.perf_counter()
                cache, rope = snapshot.restore(args.device)
                model.model.rope_deltas = rope
                _sync(args.device)
                restore_seconds = time.perf_counter() - restore_started
                reused = snapshot.tokens
                prefix_meta.update(status='exact_token_prefix_restored', prefix_fingerprint=snapshot.prefix_fingerprint,
                                   prefill_execution='resumed_candidate')
                output = model(input_ids=inputs.input_ids[:, reused:], past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            elif store is not None and boundary >= store.minimum_tokens:
                trace.phase = 'checkpoint_materialization'
                checkpoint_started = time.perf_counter()
                saved_rope = getattr(model.model, 'rope_deltas', None)
                prefix_inputs = {name: value[:, :boundary] for name, value in inputs.items()
                                 if name in {'input_ids', 'attention_mask', 'mm_token_type_ids'}}
                output = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
                _sync(args.device)
                checkpoint_prefill_seconds = time.perf_counter() - checkpoint_started
                copy_started = time.perf_counter()
                try:
                    checkpoint = PrefixStateSnapshot(output.past_key_values, prefix_inputs['input_ids'],
                                                     getattr(model.model, 'rope_deltas', None),
                                                     capacity_bytes=store.capacity_bytes)
                    admitted = store.admit(checkpoint, contract=contract)
                    prefix_meta.update(status='checkpoint_created' if admitted else 'checkpoint_not_admitted',
                                       snapshot_bytes=checkpoint.bytes)
                    del checkpoint
                except ValueError as error:
                    prefix_meta.update(status='capacity_recompute', detail=str(error))
                _sync(args.device)
                snapshot_seconds = time.perf_counter() - copy_started
                del output
                model.model.rope_deltas = saved_rope
                trace.phase = 'prefill'
                output = model(**inputs, use_cache=True, logits_to_keep=1)
            else:
                output = model(**inputs, use_cache=True, logits_to_keep=1)
            _sync(args.device)
            prefill_seconds = time.perf_counter() - prefill_started
            ttft = time.perf_counter() - started
            logits = output.logits[:, -1, :].float().clone()
            trace.phase = 'decode'
            decode_started = time.perf_counter()
            answer, generated, stop_reason = greedy_decode(model, output, processor.tokenizer,
                                                            args.gen_tokens, args.device)
            _sync(args.device)
            decode_seconds = time.perf_counter() - decode_started
            serving_seconds = time.perf_counter() - started
            candidate_execution = {'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
                                   'generated_tokens': len(generated), 'stop_reason': stop_reason,
                                   'ttft_from_reader_start_seconds': ttft}
            del output
            if snapshot is None:
                validation = {'status': 'reference_execution', 'overhead_seconds': 0.0,
                              'reference_path_executed': True, 'candidate_reuse_executed': False,
                              'full_greedy_token_ids_equal': None, 'stop_reason_equal': None,
                              'scope': 'submitted unsplit full-prefill greedy reference; no equivalence comparison',
                              'output_token_ids_sha256': hashlib.sha256(json.dumps(generated).encode()).hexdigest()}
            if validate_request:
                trace.phase = 'validation'
                validation_started = time.perf_counter()
                reference = model(**inputs, use_cache=True, logits_to_keep=1)
                _sync(args.device)
                reference_prefill = time.perf_counter() - validation_started
                difference = float((logits - reference.logits[:, -1, :].float()).abs().max())
                reference_ttft = time.perf_counter() - started
                reference_decode_started = time.perf_counter()
                reference_answer, reference_ids, reference_stop = greedy_decode(
                    model, reference, processor.tokenizer, args.gen_tokens, args.device)
                _sync(args.device)
                elapsed = time.perf_counter() - validation_started
                reference_decode_seconds = time.perf_counter() - reference_decode_started
                same_output = generated == reference_ids
                same_stop = stop_reason == reference_stop
                passed = difference <= 0.5 and same_output and same_stop
                validation = {'status': 'pass' if passed else 'fail', 'seconds': elapsed,
                              'reference_full_prefill_seconds': reference_prefill,
                              'full_greedy_token_ids_equal': same_output,
                              'stop_reason_equal': same_stop,
                              'max_absolute_logit_difference': difference, 'absolute_tolerance': 0.5,
                              'scope': 'first prefill logits and complete greedy output for this question',
                              'overhead_seconds': elapsed}
                if not passed:
                    answer, generated, stop_reason = reference_answer, reference_ids, reference_stop
                    prefix_meta['status'] = 'numeric_contract_full_recompute'
                    prefix_meta['prefill_execution'] = 'resumed_candidate_reference_fallback'
                    prefix_meta['rejected_reused_tokens'] = reused
                    reused = 0
                    prefill_seconds = reference_prefill
                    decode_seconds = reference_decode_seconds
                    ttft = reference_ttft
                    validation['overhead_seconds'] = 0.0
                    validation['candidate_execution_rejected'] = candidate_execution
                    validation['reference_consumed_as_fallback'] = True
                    serving_seconds = time.perf_counter() - started
                    if store is not None:
                        store.invalidate()
                del reference
        prefix_meta.update(reused_tokens=reused, snapshot_seconds=snapshot_seconds, restore_seconds=restore_seconds,
                           checkpoint_prefill_seconds=checkpoint_prefill_seconds,
                           store=store.metrics() if store is not None else None,
                           memory_checkpoint_boundary=boundary,
                           numerical_scope=validation.get('scope', 'not_validated'),
                           diagnostic_net_prefill_seconds=(validation['reference_full_prefill_seconds'] - prefill_seconds
                                                           if validation.get('status') == 'pass' else None),
                           diagnostic_comparison_warmed_reference=True,
                           accounting='prefill_seconds already includes snapshot and restore; allocation costs paid on creating request')
        return {'answer': answer, 'first_answer': answer, 'proof_answer': '',
                'verification_mode': 'single_pass_submitted_answer', 'submitted_answer_source': 'single_pass',
                'execution_stages': {'first': {'prompt_tokens': len(first_ids), 'generated_tokens': len(generated),
                                             'causal_kv_reused_tokens': reused, 'residual_prefill_tokens': len(first_ids) - reused,
                                             'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
                                             'generation_limit_reached': stop_reason == 'max_new_tokens',
                                             'stop_reason': stop_reason, 'ttft_from_reader_start_seconds': ttft},
                                     'proof': {'prompt_tokens': 0, 'generated_tokens': 0, 'prefill_seconds': 0.0, 'decode_seconds': 0.0,
                                               'generation_limit_reached': False, 'stop_reason': 'not_executed'}},
                'prefix_state_reuse': prefix_meta, 'prefix_validation': validation,
                'submitted_output_token_ids_sha256': hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                'numeric_audit_scheduled': validate_request,
                'serialized_memory_tokens': memory_tokens, 'reader_seconds': time.perf_counter() - started,
                'single_pass_serving_excluding_reference_seconds': serving_seconds,
                'operator_trace': trace.as_dict()}
    finally:
        trace.close()
