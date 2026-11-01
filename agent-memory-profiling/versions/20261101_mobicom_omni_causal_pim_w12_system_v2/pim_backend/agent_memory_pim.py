"""Trajectory-aware LPDDR-PIM admission and residual execution planning."""
from __future__ import annotations

import math
import numpy as np

from .lpddr_bank_pim import bank_sharded_attention


def plan_materialized_scan(backend, identities, dimension, top_k, resident) -> dict:
    identities = tuple(identities)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('scan requires distinct immutable vector identities')
    if len(resident) * dimension > backend.config.pim_capacity_bytes:
        raise ValueError('resident state exceeds PIM capacity')
    host = backend.resident_score_scan(len(identities), dimension, top_k=top_k, route='host')
    pim = backend.resident_score_scan(len(identities), dimension, top_k=top_k, route='pim')
    missing = set(identities) - resident
    cold = backend.copy_cost(len(missing) * dimension)
    copy_control = (backend.refresh_elapsed(backend.config.host_command_cycles +
                                            backend.config.host_fence_cycles) if missing else 0)
    cold_cycles = cold['cycles'] + copy_control
    pim_admitted = pim.get('route') == 'pim' and pim.get('admission', {}).get('admitted', True)
    pim_total = pim['cycles'] + cold_cycles
    saving_per_scan = host['cycles'] - pim['cycles']
    break_even = math.floor(cold_cycles / saving_per_scan) + 1 if saving_per_scan > 0 else None
    selected = 'pim' if pim_admitted and pim_total < host['cycles'] else 'host'
    evicted = len(resident - set(identities)) if selected == 'pim' else 0
    if selected == 'pim':
        resident.intersection_update(identities)
        resident.update(identities)
    return {
        'policy': 'current_known_scan_cost_only_v1', 'known_dag_references': 1,
        'selected_route': selected, 'host_cycles': host['cycles'],
        'pim_cycles': pim['cycles'], 'cold_copy_cycles': cold_cycles,
        'copy_command_fence_cycles': copy_control,
        'selected_cycles': pim_total if selected == 'pim' else host['cycles'],
        'break_even_references': break_even, 'pim_cycles_including_copy': pim_total,
        'modeled_saving_cycles': host['cycles'] - (pim_total if selected == 'pim' else host['cycles']),
        'missing_resident_vectors': len(missing), 'new_resident_bytes': len(missing) * dimension if selected == 'pim' else 0,
        'hypothetical_cold_copy_bytes': len(missing) * dimension,
        'cold_copy_bytes': len(missing) * dimension if selected == 'pim' else 0,
        'evicted_immutable_bytes': evicted * dimension, 'eviction_writeback_bytes': 0,
        'resident_bytes': len(resident) * dimension, 'capacity_bytes': backend.config.pim_capacity_bytes,
        'dimension': dimension, 'vectors': len(identities), 'top_k': top_k,
        'admission': pim['admission'],
        'fallback_reason': (None if selected == 'pim' else 'capacity_or_workspace' if not pim_admitted
                            else 'cold_copy_not_amortized' if saving_per_scan > 0 else 'host_scan_cheaper'),
        'host_trace': host, 'pim_trace': pim, 'cold_copy_trace': cold,
        'functional_contract': 'bank_int8_topk checked against int32 reference by caller',
        'boundary': 'one current scan; no assumed future query reuse; uncalibrated simulation, not device wall time',
    }


def plan_attention_route(backend, context_tokens: int, *, query_heads: int,
                         kv_heads: int, head_dim: int, resident_bytes: int,
                         element_bytes: int = 2, cold_bytes: int = 0,
                         horizon_steps: int = 1) -> dict:
    """Choose a route for the residual GQA attention primitive.

    This is the missing bridge between retrieval-side PIM and the Reader's
    memory-bound attention phase.  It remains a model-level cost decision;
    the caller must provide a device-calibrated profile before claiming E2E.
    """
    required_bytes = 2 * context_tokens * kv_heads * head_dim * element_bytes
    if horizon_steps < 1:
        raise ValueError('positive non-oracle horizon required')
    if cold_bytes < 0 or cold_bytes > required_bytes:
        raise ValueError('cold bytes must be within this layer KV state')
    if resident_bytes < required_bytes:
        raise ValueError('resident_bytes must include both complete K and V tensors')
    admission = backend.admission(resident_bytes=resident_bytes, query_heads=query_heads,
                                  head_dim=head_dim, element_bytes=element_bytes)
    host = backend.attention_step(context_tokens, query_heads=query_heads, kv_heads=kv_heads,
                                  head_dim=head_dim, element_bytes=element_bytes, route='host')
    pim = backend.attention_step(context_tokens, query_heads=query_heads, kv_heads=kv_heads,
                                 head_dim=head_dim, element_bytes=element_bytes, route='pim')
    cold_cycles = backend.copy_cost(cold_bytes)['cycles'] if cold_bytes else 0
    pim_total = pim['cycles'] + cold_cycles
    append_cycles = backend.copy_cost(2 * kv_heads * head_dim * element_bytes)['cycles']
    host_horizon = horizon_steps * host['cycles']
    pim_horizon = cold_cycles + horizon_steps * pim['cycles'] + (horizon_steps - 1) * append_cycles
    step_saving = host['cycles'] - pim['cycles'] - append_cycles
    break_even = max(1, math.floor((cold_cycles - append_cycles) / step_saving) + 1) if step_saving > 0 else None
    selected = 'pim' if admission['admitted'] and pim_horizon < host_horizon else 'host'
    return {'selected_route': selected, 'host_cycles': host['cycles'], 'pim_cycles': pim['cycles'],
            'cold_copy_bytes': cold_bytes, 'cold_copy_cycles': cold_cycles,
            'pim_cycles_including_copy': pim_total,
            'selected_cycles': pim_total if selected == 'pim' else host['cycles'],
            'forecast_steps': horizon_steps, 'break_even_steps': break_even,
            'host_horizon_cycles': host_horizon, 'pim_horizon_cycles': pim_horizon,
            'append_copy_cycles_per_step': append_cycles,
            'modeled_saving_cycles': host['cycles'] - pim_total, 'admission': admission,
            'host_trace': host, 'pim_trace': pim,
            'boundary': 'GQA attention route model; no device wall-clock or energy claim'}


def linear_kernel_cost(backend, *, rows, in_features, out_features, element_bytes, available_bytes):
    if min(rows, in_features, out_features, element_bytes) < 1 or available_bytes < 0:
        raise ValueError('invalid linear shape or capacity')
    cfg = backend.config
    weights = in_features * out_features * element_bytes
    io_bytes = rows * (in_features + out_features) * element_bytes
    macs = rows * in_features * out_features
    host_active = cfg.host_command_cycles + max(backend.transfer_cycles(weights + io_bytes),
                                               math.ceil(macs / cfg.host_macs_per_cycle))
    host = backend.refresh_elapsed(host_active)
    rows_per_bank = math.ceil(out_features / cfg.bank_count)
    workspace = (in_features + rows_per_bank) * element_bytes
    admitted = rows == 1 and weights <= available_bytes and workspace <= cfg.local_buffer_bytes_per_bank
    counts = tuple(out_features // cfg.bank_count + int(bank < out_features % cfg.bank_count)
                   for bank in range(cfg.bank_count))
    memory, activations = backend._bank_memory(counts, in_features * element_bytes)
    pim_active = (2 * cfg.host_command_cycles + cfg.host_fence_cycles + memory
                  + math.ceil(rows_per_bank * in_features / cfg.pim_macs_per_bank_cycle)
                  + backend.transfer_cycles(cfg.bank_count * in_features * element_bytes
                                            + out_features * element_bytes))
    warm = backend.refresh_elapsed(pim_active)
    cold = backend.copy_cost(weights)['cycles']
    return {'host_cycles': host, 'pim_resident_cycles': warm, 'cold_write_cycles': cold,
            'pim_cold_cycles': warm + cold, 'weight_bytes': weights, 'macs': macs,
            'workspace_per_bank': workspace, 'row_activations': activations,
            'admitted': admitted, 'rows': rows,
            'selected_cold_route': 'pim' if admitted and warm + cold < host else 'host',
            'selected_warm_route': 'pim' if admitted and warm < host else 'host',
            'boundary': 'row-sharded single-token GEMV assumption; GEMM host-only; bias/gate/norm excluded'}


def replay_reader_residual(backend, text_config, result):
    layer_types = list(text_config.layer_types)
    full_layers = layer_types.count('full_attention')
    heads = text_config.num_attention_heads
    kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim
    first = result['execution_stages']['first']
    prompt_tokens = first['prompt_tokens']
    decode_calls = max(0, first['generated_tokens'] - 1)
    retrieval_bytes = result.get('pim_retrieval', {}).get('resident_bytes', 0)
    resident_context = 0
    attention = []
    for offset in range(decode_calls if full_layers else 0):
        context = prompt_tokens + offset + 1
        total_kv_bytes = full_layers * 2 * context * kv_heads * head_dim * 2
        route = plan_attention_route(backend, context, query_heads=heads, kv_heads=kv_heads,
                                     head_dim=head_dim, resident_bytes=total_kv_bytes + retrieval_bytes,
                                     cold_bytes=2 * (context - resident_context) * kv_heads * head_dim * 2,
                                     horizon_steps=1)
        if route['selected_route'] == 'pim':
            resident_context = context
        route.pop('host_trace')
        route.pop('pim_trace')
        route.update(context_tokens=context, layer_multiplicity=full_layers)
        attention.append(route)
    peak_context = prompt_tokens + decode_calls
    kv_bytes = full_layers * 2 * peak_context * kv_heads * head_dim * 2
    linears = []
    for record in result.get('operator_trace', {}).get('operators', []):
        if record['phase'] not in {'prefill', 'decode'}:
            continue
        cost = linear_kernel_cost(backend, rows=record['rows'], in_features=record['in_features'],
                                  out_features=record['out_features'], element_bytes=record['element_bytes'],
                                  available_bytes=max(0, backend.config.pim_capacity_bytes - kv_bytes - retrieval_bytes))
        linears.append({**record, **cost})
    return {'layer_types': dict((kind, layer_types.count(kind)) for kind in sorted(set(layer_types))),
            'attention_decode_steps': attention, 'linear_calls': linears,
            'peak_full_attention_kv_bytes': kv_bytes, 'pool_capacity_bytes': backend.config.pim_capacity_bytes,
            'retrieval_resident_bytes': retrieval_bytes,
            'modeled_attention_host_cycles': sum(step['host_cycles'] * full_layers for step in attention),
            'modeled_attention_selected_cycles': sum(step['selected_cycles'] * full_layers for step in attention),
            'attention_residency_boundary': 'route sees only the current decode primitive; no output-length or future-reuse forecast; cold-copy and append included; per-layer costs multiplied by full-attention layers; simulation only',
            'prefill_residual_tokens': first['residual_prefill_tokens'],
            'submitted_path_numeric_fallback': result.get('prefix_validation', {}).get('status') == 'fail',
            'trace_complete_for_submitted_path': result.get('prefix_validation', {}).get('status') != 'fail',
            'boundary': 'trace-driven primitive envelopes, not a scheduled full-model speedup; warm linears are mutually exclusive residency scenarios; linear recurrent state and model/feature memory still require device allocation',
            'hardware_output_substitution': False}


def validate_attention_functional(query: np.ndarray, keys: np.ndarray, values: np.ndarray,
                                  *, banks: int, tolerance: float = 1e-5) -> dict:
    """Check bank-sharded attention against an unsharded NumPy reference."""
    query = np.asarray(query, dtype=np.float32)
    keys = np.asarray(keys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    actual = bank_sharded_attention(query, keys, values, banks=banks)
    group = query.shape[0] // keys.shape[1]
    head_map = np.arange(query.shape[0]) // group
    scores = np.einsum('hd,nhd->hn', query, keys[:, head_map]) / np.sqrt(query.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    reference = np.einsum('hn,nhd->hd', probabilities, values[:, head_map])
    max_error = float(np.max(np.abs(actual - reference)))
    return {'exact_within_tolerance': max_error <= tolerance, 'max_absolute_error': max_error,
            'tolerance': tolerance, 'banks': banks, 'context_tokens': int(keys.shape[0]),
            'query_heads': int(query.shape[0]), 'kv_heads': int(keys.shape[1])}
