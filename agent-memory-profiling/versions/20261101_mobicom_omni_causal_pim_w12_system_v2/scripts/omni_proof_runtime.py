"""Frozen Reader execution and source identities for the recovery candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext

from stage_profiler import allowed_token_count, summarize_grammar_allowed


def checksum(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def parse_object(response):
    text = response.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON object key: ' + key)
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError('one JSON object required')
    return value


def authorize_development_users(prepared, users):
    return authorize_users(prepared, users, allowed_splits={'development'})


def authorize_users(prepared, users, *, allowed_splits):
    prepared = Path(prepared)
    audit = json.loads((prepared / 'audit.json').read_text())
    if file_digest(prepared / 'split.json') != audit['artifacts_sha256']['split.json']:
        raise ValueError('serving split identity changed')
    split = json.loads((prepared / 'split.json').read_text())
    people_by_split = {
        'development': set(split.get('development_people', [])),
        'validation': set(split.get('validation_people', [])),
        'holdout': set(split.get('holdout_people', [])),
    }
    allowed_people = set().union(*(people_by_split[name] for name in allowed_splits))
    if not users or not set(users) <= allowed_people:
        if allowed_splits == {'development'}:
            raise ValueError('development users only; validation and holdout remain sealed')
        raise ValueError(f'users are outside the authorized splits: {sorted(set(users) - allowed_people)}')
    return audit


def authorize_history(prepared, users):
    prepared = Path(prepared)
    audit = authorize_development_users(prepared, users)
    return _load_authorized_history(prepared, users, audit)


def authorize_history_users(prepared, users, *, allowed_splits):
    prepared = Path(prepared)
    audit = authorize_users(prepared, users, allowed_splits=allowed_splits)
    return _load_authorized_history(prepared, users, audit)


def _load_authorized_history(prepared, users, audit):
    if file_digest(prepared / 'omni_graph.json') != audit['artifacts_sha256']['omni_graph.json']:
        raise ValueError('serving source identity changed')
    return [graph for graph in json.loads((prepared / 'omni_graph.json').read_text())
            if graph['person_id'] in users]


def load_reader(path, pixels=262144):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from omni_patch_projection import configure_patch_projection

    torch.set_num_threads(2)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size['longest_edge'] = pixels
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:0').eval()
    configure_patch_projection(model.model.visual, 'linear')
    return processor, model


def prepare_inputs(processor, messages, images=()):
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    arguments = {'text': [rendered], 'return_tensors': 'pt'}
    if images:
        arguments['images'] = list(images)
    return processor(**arguments)


def _profiled_operator_events(profiler, *, phase, profiled_forward_calls):
    events = []
    for event in profiler.key_averages(group_by_input_shape=True):
        device_time = float(getattr(event, 'device_time_total', 0.0))
        cpu_time = float(getattr(event, 'cpu_time_total', 0.0))
        if device_time <= 0 and cpu_time <= 0:
            continue
        events.append({
            'operator': str(event.key),
            'calls': int(event.count),
            'input_shapes': event.input_shapes,
            'inclusive_device_microseconds': device_time,
            'self_device_microseconds': float(getattr(event, 'self_device_time_total', 0.0)),
            'inclusive_cpu_microseconds': cpu_time,
            'self_cpu_microseconds': float(getattr(event, 'self_cpu_time_total', 0.0)),
        })
    events.sort(key=lambda item: (-item['inclusive_device_microseconds'], item['operator']))
    return {
        'phase': phase,
        'profiled_model_forward_calls': profiled_forward_calls,
        'operator_events': events,
        'timing_scope': 'torch.profiler operator activity; inclusive event times may overlap',
    }


def greedy(model, processor, inputs, *, max_tokens, trace=None, grammar_matcher=None,
           profile_operator_kernels=False, profile_decode_forward_calls=4,
           profile_reader=False, stage_profiler=None, stage_prefix=None,
           n_images=0, lm_head_shadow=None, lm_head_shadow_phase=None):
    import torch

    if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 0:
        raise ValueError('n_images must be a non-negative integer')
    if stage_profiler is not None and not stage_prefix:
        raise ValueError('stage_prefix is required with a stage profiler')
    if lm_head_shadow is not None and not lm_head_shadow_phase:
        raise ValueError('lm_head shadow phase is required when shadowing')
    profiling = bool(profile_reader or stage_profiler is not None)
    started = time.perf_counter()
    inputs = inputs.to('cuda:0')
    torch.cuda.synchronize()
    input_tokens = int(inputs.input_ids.shape[1])
    valid_vocab_size = len(processor.tokenizer)
    if trace is not None:
        trace.phase = 'prefill'
    prefill_profiler = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True) if profile_operator_kernels else nullcontext())
    prefill_stage = (stage_profiler.stage(f'{stage_prefix}_prefill')
                     if stage_profiler is not None else nullcontext(None))
    with prefill_stage as prefill_span:
        prefill_forward_started = time.perf_counter()
        with prefill_profiler as active_prefill_profiler:
            with torch.inference_mode():
                output = model(**inputs, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
        prefill_forward_seconds = time.perf_counter() - prefill_forward_started
        if prefill_span is not None:
            prefill_span.set_metrics({
                'prefill_tokens': input_tokens,
                'n_images': n_images,
                'decode_tokens': 0,
                'grammar_forced_tokens': 0,
                'grammar_allowed_mean': None,
            })
    prefill = time.perf_counter() - started
    prefill_live_inputs = [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype),
        'logical_bytes': value.numel() * value.element_size(), 'consumer': 'prefill'}
        for name, value in inputs.items() if isinstance(value, torch.Tensor)]
    released = []
    for name in ('pixel_values', 'image_grid_thw'):
        if name in inputs:
            released.append(name)
            del inputs[name]
    cache = output.past_key_values
    stops = getattr(model.config.text_config, 'eos_token_id', processor.tokenizer.eos_token_id)
    stops = set(stops if isinstance(stops, list) else [stops])
    tokens, reason, first_token_seconds = [], 'max_new_tokens', None
    if trace is not None:
        trace.phase = 'decode'
    decode_timing = ({
        'logit_slice_seconds': 0.0,
        'grammar_fill_seconds': 0.0,
        'grammar_apply_seconds': 0.0,
        'argmax_launch_seconds': 0.0,
        'token_sync_seconds': 0.0,
        'grammar_accept_seconds': 0.0,
        'forward_launch_seconds': 0.0,
        'loop_total_seconds': 0.0,
    } if trace is not None else None)
    grammar_mask = None
    grammar_allowed_counts = [] if profiling else None
    profiled_decode_forwards = 0
    decode_profiler = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(wait=0, warmup=0,
                                         active=profile_decode_forward_calls, repeat=1))
        if profile_operator_kernels else nullcontext())
    decode_stage = (stage_profiler.stage(f'{stage_prefix}_decode')
                    if stage_profiler is not None else nullcontext(None))
    with decode_stage as decode_span:
        decode_started = time.perf_counter()
        with decode_profiler as active_decode_profiler:
            for position in range(max_tokens):
                loop_started = time.perf_counter()
                stage_started = loop_started
                scores = output.logits[:, -1, :]
                if decode_timing is not None:
                    now = time.perf_counter()
                    decode_timing['logit_slice_seconds'] += now - stage_started
                    stage_started = now
                if grammar_matcher is not None:
                    import xgrammar as xgr

                    if grammar_mask is None:
                        grammar_mask = xgr.allocate_token_bitmask(1, scores.shape[-1])
                    needs_mask = grammar_matcher.fill_next_token_bitmask(grammar_mask)
                    if decode_timing is not None:
                        now = time.perf_counter()
                        decode_timing['grammar_fill_seconds'] += now - stage_started
                        stage_started = now
                    if needs_mask:
                        xgr.apply_token_bitmask_inplace(scores, grammar_mask.to(scores.device),
                                                        vocab_size=scores.shape[-1])
                    if decode_timing is not None:
                        now = time.perf_counter()
                        decode_timing['grammar_apply_seconds'] += now - stage_started
                        stage_started = now
                    if profiling:
                        grammar_allowed_counts.append(
                            allowed_token_count(grammar_mask, scores.shape[-1],
                                                valid_vocab_size)
                            if needs_mask else valid_vocab_size)
                elif profiling:
                    grammar_allowed_counts.append(valid_vocab_size)
                shadow_allowed = None
                shadow_forced = False
                shadow_scheduled = False
                if lm_head_shadow is not None:
                    if grammar_matcher is not None:
                        shadow_allowed = lm_head_shadow.allowed_from_bitmask(
                            grammar_mask)
                        shadow_forced = int(shadow_allowed.sum().item()) == 1
                        shadow_scheduled = lm_head_shadow.should_validate(
                            position, shadow_forced)
                    else:
                        shadow_scheduled = lm_head_shadow.should_validate(
                            position, False)
                        if shadow_scheduled:
                            shadow_allowed = lm_head_shadow.full_allowed()
                token = scores.argmax(-1, keepdim=True)
                if decode_timing is not None:
                    now = time.perf_counter()
                    decode_timing['argmax_launch_seconds'] += now - stage_started
                    stage_started = now
                token_value = int(token.item())
                tokens.append(token_value)
                if shadow_scheduled:
                    lm_head_shadow.observe(
                        phase=lm_head_shadow_phase,
                        actual_token=token_value,
                        allowed=shadow_allowed,
                        forced=shadow_forced)
                if decode_timing is not None:
                    now = time.perf_counter()
                    decode_timing['token_sync_seconds'] += now - stage_started
                    stage_started = now
                if grammar_matcher is not None and not grammar_matcher.accept_token(tokens[-1]):
                    raise RuntimeError('constrained JSON decoder rejected its selected token')
                if decode_timing is not None:
                    now = time.perf_counter()
                    decode_timing['grammar_accept_seconds'] += now - stage_started
                    stage_started = now
                if position == 0:
                    first_token_seconds = time.perf_counter() - started
                if grammar_matcher is not None and grammar_matcher.is_terminated():
                    reason = 'json_complete'
                    break
                if tokens[-1] in stops:
                    reason = 'eos'
                    break
                if position + 1 < max_tokens:
                    with torch.inference_mode():
                        output = model(input_ids=token, past_key_values=cache,
                                       use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    if decode_timing is not None:
                        now = time.perf_counter()
                        decode_timing['forward_launch_seconds'] += now - stage_started
                    if (active_decode_profiler is not None
                            and profiled_decode_forwards < profile_decode_forward_calls):
                        profiled_decode_forwards += 1
                        active_decode_profiler.step()
                if decode_timing is not None:
                    decode_timing['loop_total_seconds'] += time.perf_counter() - loop_started
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_started
        grammar_summary = (summarize_grammar_allowed(grammar_allowed_counts)
                           if profiling else {'grammar_forced_tokens': 0,
                                              'grammar_allowed_mean': None})
        grammar_forced_tokens = grammar_summary['grammar_forced_tokens']
        grammar_allowed_mean = grammar_summary['grammar_allowed_mean']
        if decode_span is not None:
            decode_span.set_metrics({
                'prefill_tokens': input_tokens,
                'n_images': n_images,
                'decode_tokens': len(tokens),
                'grammar_forced_tokens': grammar_forced_tokens,
                'grammar_allowed_mean': grammar_allowed_mean,
            })
    operator_profile = None
    if profile_operator_kernels:
        operator_profile = {
            'prefill': _profiled_operator_events(
                active_prefill_profiler, phase='prefill', profiled_forward_calls=1),
            'decode': _profiled_operator_events(
                active_decode_profiler, phase='decode',
                profiled_forward_calls=profiled_decode_forwards),
            'performance_comparable': False,
            'boundary': ('diagnostic-only CUDA operator activity; profiler overhead is included in '
                         'serving wall times and event durations are not target PIM latency'),
        }
    if decode_timing is not None:
        accounted = sum(value for key, value in decode_timing.items()
                        if key != 'loop_total_seconds')
        decode_timing['residual_seconds'] = max(0.0, decode_timing['loop_total_seconds'] - accounted)
        decode_timing['generated_tokens'] = len(tokens)
        decode_timing['boundary'] = ('per-token Host-side stage timing; token-sync includes dependent '
                                     'GPU work, while forward-launch is asynchronous issue time')
    else:
        decode_timing = None
    result = {'answer': processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
            'answer_token_ids': tokens, 'stop_reason': reason,
            'prefill_seconds': prefill, 'decode_seconds': decode,
            'decode_phase_profile': decode_timing,
            'input_tokens': input_tokens, 'prefill_input_allocations': prefill_live_inputs,
            'released_before_decode': released,
            'first_token_from_reader_start_seconds': first_token_seconds,
            'reader_seconds': time.perf_counter() - started,
            'operator_profile': operator_profile}
    if profiling:
        result['reader_stage_profile'] = {
            'prefill_wall_ms': prefill_forward_seconds * 1000.0,
            'decode_wall_ms': decode * 1000.0,
            'prefill_tokens': input_tokens,
            'n_images': n_images,
            'decode_tokens': len(tokens),
            'grammar_forced_tokens': grammar_forced_tokens,
            'grammar_allowed_mean': grammar_allowed_mean,
        }
    return result


def structured_call(model, processor, system, payload, *, max_tokens=512, budget=8192,
                    json_schema=None, trace=None, profile_operator_kernels=False,
                    profile_decode_forward_calls=4, profile_reader=False,
                    stage_profiler=None, stage_prefix=None, n_images=0,
                    lm_head_shadow=None, lm_head_shadow_phase=None):
    started = time.perf_counter()
    prepare_started = time.perf_counter()
    inputs = prepare_inputs(processor, [{'role': 'system', 'content': system},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
    prepare_seconds = time.perf_counter() - prepare_started
    if inputs.input_ids.shape[1] > budget:
        raise ValueError('complete structured input exceeds frozen context; no truncation')
    matcher = None
    grammar_compile_seconds = 0.0
    if json_schema is not None:
        import xgrammar as xgr

        tokenizer = processor.tokenizer
        vocab_size = int(model.config.text_config.vocab_size)
        cache_key = (id(tokenizer), vocab_size, json.dumps(json_schema, sort_keys=True))
        grammar = _JSON_GRAMMAR_CACHE.get(cache_key)
        if grammar is None:
            grammar_started = time.perf_counter()
            stop_tokens = getattr(model.config.text_config, 'eos_token_id', tokenizer.eos_token_id)
            stop_tokens = stop_tokens if isinstance(stop_tokens, list) else [stop_tokens]
            info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size,
                                                       stop_token_ids=stop_tokens)
            compiler = xgr.GrammarCompiler(info, max_threads=2)
            grammar = compiler.compile_json_schema(json_schema, strict_mode=True, separators=(',', ':'))
            _JSON_GRAMMAR_CACHE[cache_key] = grammar
            grammar_compile_seconds = time.perf_counter() - grammar_started
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    result = greedy(model, processor, inputs, max_tokens=max_tokens, trace=trace,
                    grammar_matcher=matcher,
                    profile_operator_kernels=profile_operator_kernels,
                    profile_decode_forward_calls=profile_decode_forward_calls,
                    profile_reader=profile_reader, stage_profiler=stage_profiler,
                    stage_prefix=stage_prefix, n_images=n_images,
                    lm_head_shadow=lm_head_shadow,
                    lm_head_shadow_phase=lm_head_shadow_phase)
    postprocess_started = time.perf_counter()
    try:
        result['parsed'] = parse_object(result['answer']) if result['stop_reason'] in {'eos', 'json_complete'} else None
        if result['parsed'] is not None and json_schema is not None:
            from jsonschema import Draft202012Validator

            if not Draft202012Validator(json_schema).is_valid(result['parsed']):
                result['parsed'] = None
    except (ValueError, TypeError):
        result['parsed'] = None
    postprocess_seconds = time.perf_counter() - postprocess_started
    result['reader_seconds'] = time.perf_counter() - started
    result['structured_input_tokens'] = int(inputs.input_ids.shape[1])
    result['structured_prepare_seconds'] = prepare_seconds
    result['structured_grammar_compile_seconds'] = grammar_compile_seconds
    result['structured_postprocess_seconds'] = postprocess_seconds
    result['structured_reader_prefill_seconds'] = result['prefill_seconds']
    result['structured_reader_decode_seconds'] = result['decode_seconds']
    result['structured_first_token_seconds'] = result['first_token_from_reader_start_seconds']
    result['structured_operator_profile'] = result.pop('operator_profile')
    return result


_JSON_GRAMMAR_CACHE = {}
