"""Label-free serving-time candidate selection for the 48-wide pool."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


CALIBRATOR_SCHEMA = 'omni-conformal-calibrator-v1'
POLICY_STATS_SCHEMA = 'omni-candidate-policy-stats-v1'
CANDIDATE_WIDTH = 48


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be a lowercase sha256')
    return value


def _canonical_sha256(payload):
    body = {key: value for key, value in payload.items() if key != 'sha256'}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_calibrator(path, expected_sha256):
    """Load a T3.2 calibrator and verify both its identity and self-hash."""
    expected_sha256 = _sha256(expected_sha256, 'expected calibrator sha256')
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError('calibrator payload must be an object')
    if payload.get('schema') != CALIBRATOR_SCHEMA:
        raise ValueError('calibrator schema mismatch')
    declared = _sha256(payload.get('sha256'), 'calibrator sha256')
    if declared != _canonical_sha256(payload):
        raise ValueError('calibrator sha256 mismatch')
    if declared != expected_sha256:
        raise ValueError('calibrator does not match configured sha256')
    if payload.get('split') != 'development':
        raise ValueError('calibrator must be fitted on development split')
    if payload.get('candidate_width') != CANDIDATE_WIDTH:
        raise ValueError('calibrator candidate width must be 48')
    tau = payload.get('tau')
    if not isinstance(tau, dict) or not tau:
        raise ValueError('calibrator tau table missing')
    for key, value in tau.items():
        if value is not None:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(f'calibrator tau must be finite or null: {key}')
    return payload


def threshold_for_alpha(calibrator, alpha):
    """Return the stored threshold, decoding JSON ``null`` as ``-inf``."""
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not 0.0 < float(alpha) < 1.0):
        raise ValueError('alpha must lie strictly between zero and one')
    key = str(float(alpha))
    tau = calibrator.get('tau', {})
    if key not in tau:
        raise ValueError(f'calibrator has no alpha threshold: {key}')
    value = tau[key]
    return -math.inf if value is None else float(value)


def _record_rank(record):
    if not isinstance(record, dict):
        raise ValueError('candidate record must be an object')
    score = record.get('score')
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score))):
        raise ValueError('candidate score must be finite numeric')
    demand_id = record.get('demand_id')
    base_record_id = record.get('base_record_id')
    if not isinstance(demand_id, str) or not demand_id:
        raise ValueError('candidate demand identity required')
    if not isinstance(base_record_id, str) or not base_record_id:
        raise ValueError('candidate source identity required')
    query_rank = record.get('query_rank', 0)
    retrieval_rank = record.get('retrieval_rank', 0)
    if (isinstance(query_rank, bool) or not isinstance(query_rank, int)
            or isinstance(retrieval_rank, bool)
            or not isinstance(retrieval_rank, int)):
        raise ValueError('candidate ranks must be integers')
    return (-float(score), query_rank, retrieval_rank, base_record_id,
            str(record.get('witness_id', '')))


def _token_count(records, counter):
    return int(counter(records)) if counter is not None else None


def apply_conformal_candidate_policy(candidates, *, demand_ids=None, tau,
                                     min_per_demand=4,
                                     candidate_width=CANDIDATE_WIDTH,
                                     alpha=0.1, calibrator_sha256=None,
                                     token_counter=None):
    """Retain score-qualified candidates per demand, then fill to the minimum."""
    if not isinstance(candidates, list):
        raise ValueError('candidates must be a list')
    if (isinstance(candidate_width, bool) or not isinstance(candidate_width, int)
            or candidate_width != CANDIDATE_WIDTH):
        raise ValueError('candidate policy requires the fixed 48-wide pool')
    if (isinstance(min_per_demand, bool) or not isinstance(min_per_demand, int)
            or not 1 <= min_per_demand <= candidate_width):
        raise ValueError('min_per_demand must lie in [1, 48]')
    if (isinstance(tau, bool) or not isinstance(tau, (int, float))
            or math.isnan(float(tau))):
        raise ValueError('candidate policy threshold must be numeric')
    tau = float(tau)
    if demand_ids is None:
        demand_ids = {
            record.get('demand_id') if isinstance(record, dict) else None
            for record in candidates
        }
    demand_ids = list(demand_ids)
    if (any(not isinstance(value, str) or not value for value in demand_ids)
            or len(set(demand_ids)) != len(demand_ids)):
        raise ValueError('demand_ids must be unique nonempty strings')
    demand_ids.sort()

    grouped = {demand_id: [] for demand_id in demand_ids}
    for record in candidates:
        demand_id = record.get('demand_id') if isinstance(record, dict) else None
        if demand_id not in grouped:
            raise ValueError(f'candidate belongs to an unknown demand: {demand_id}')
        _record_rank(record)
        grouped[demand_id].append(record)

    selected_records = []
    stats_by_demand = []
    for demand_id in demand_ids:
        ordered = sorted(grouped[demand_id], key=_record_rank)
        if len(ordered) > candidate_width:
            raise ValueError(f'demand exceeds the fixed candidate width: {demand_id}')
        retained = [record for record in ordered if float(record['score']) >= tau]
        retained = retained[:candidate_width]
        selected = list(retained)
        if len(selected) < min_per_demand:
            selected_ids = {id(record) for record in selected}
            for record in ordered:
                if id(record) in selected_ids:
                    continue
                selected.append(record)
                selected_ids.add(id(record))
                if len(selected) >= min_per_demand:
                    break
        selected = selected[:candidate_width]
        before_tokens = _token_count(ordered, token_counter)
        after_tokens = _token_count(selected, token_counter)
        stats_by_demand.append({
            'demand_id': demand_id,
            'candidate_count_before': len(ordered),
            'candidate_count_after': len(selected),
            'retained_by_threshold': len(retained),
            'filled_to_min': len(selected) - min(len(retained), len(selected)),
            'candidate_count_shortfall': max(0, min_per_demand - len(ordered)),
            'witness_candidate_tokens_before': before_tokens,
            'witness_candidate_tokens_after': after_tokens,
            'witness_candidate_token_delta': (
                after_tokens - before_tokens
                if before_tokens is not None and after_tokens is not None
                else None),
        })
        selected_records.extend(selected)

    before_total = sum(item['candidate_count_before'] for item in stats_by_demand)
    after_total = sum(item['candidate_count_after'] for item in stats_by_demand)
    token_before_total = (
        sum(item['witness_candidate_tokens_before'] for item in stats_by_demand)
        if all(item['witness_candidate_tokens_before'] is not None
               for item in stats_by_demand) else None)
    token_after_total = (
        sum(item['witness_candidate_tokens_after'] for item in stats_by_demand)
        if all(item['witness_candidate_tokens_after'] is not None
               for item in stats_by_demand) else None)
    stats = {
        'schema': POLICY_STATS_SCHEMA,
        'policy': 'conformal',
        'alpha': float(alpha),
        'tau': (None if math.isinf(tau) and tau < 0 else tau),
        'tau_encoding': 'JSON null denotes negative infinity',
        'calibrator_sha256': calibrator_sha256,
        'candidate_width': candidate_width,
        'min_per_demand': min_per_demand,
        'demands': stats_by_demand,
        'candidate_count_before': before_total,
        'candidate_count_after': after_total,
        'witness_candidate_tokens_before': token_before_total,
        'witness_candidate_tokens_after': token_after_total,
        'witness_candidate_token_delta': (
            token_after_total - token_before_total
            if token_before_total is not None and token_after_total is not None
            else None),
        'token_count_boundary': (
            'compact candidate JSON token count; excludes witness prompt and demand metadata'),
    }
    return selected_records, stats
