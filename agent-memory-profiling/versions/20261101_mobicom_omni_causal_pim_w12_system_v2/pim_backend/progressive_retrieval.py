"""Progressive plane-major retrieval with v1 candidate-width semantics."""
from __future__ import annotations

import numpy as np

from .progressive_topk import (PlaneMajorMatrix, progressive_topk,
                               reference_scores)


CANDIDATE_WIDTH = 48
PRIMARY_WIDTH = 32
OVERALL_WIDTH = 80


def _allowed_mask(row_mask, total):
    if row_mask is None:
        return np.ones(total, dtype=bool)
    values = np.asarray(row_mask)
    if values.dtype == np.bool_:
        if values.shape != (total,):
            raise ValueError('row mask must match rows')
        return values.copy()
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('rows must be integer indices or a boolean mask')
    result = np.zeros(total, dtype=bool)
    indices = values.astype(np.int64, copy=False)
    if (np.any(indices < 0) or np.any(indices >= total)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('unique in-range row indices required')
    result[indices] = True
    return result


def candidate_order_v1(scores, *, required_kind_mask=None, row_mask=None,
                       candidate_width=CANDIDATE_WIDTH,
                       primary_width=PRIMARY_WIDTH):
    """Apply the v1 32-primary/48-total rule with deterministic ties by index."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('one score row required')
    if (isinstance(candidate_width, bool) or not isinstance(candidate_width, int)
            or candidate_width < 1
            or isinstance(primary_width, bool) or not isinstance(primary_width, int)
            or primary_width < 1):
        raise ValueError('positive candidate and primary widths required')
    allowed = _allowed_mask(row_mask, len(values)) & np.isfinite(values)
    if required_kind_mask is not None:
        kind = _allowed_mask(required_kind_mask, len(values))
        primary_mask = allowed & kind
    else:
        primary_mask = None

    overall = np.flatnonzero(allowed)
    overall = overall[np.argsort(-values[overall], kind='stable')]
    if primary_mask is None:
        return overall[:candidate_width].astype(np.int64, copy=False)

    primary = np.flatnonzero(primary_mask)
    primary = primary[np.argsort(-values[primary], kind='stable')]
    primary = primary[:min(primary_width, candidate_width)]
    order = [int(index) for index in primary]
    seen = set(order)
    for index in overall:
        value = int(index)
        if value not in seen:
            order.append(value)
            seen.add(value)
        if len(order) >= candidate_width:
            break
    return np.asarray(order[:candidate_width], dtype=np.int64)


def quantized_candidate_order(matrix, query, *, required_kind_mask=None,
                              row_mask=None, candidate_width=CANDIDATE_WIDTH,
                              primary_width=PRIMARY_WIDTH):
    """Compute exact float64 quantized scores, then run the v1 selection rule."""
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    allowed = _allowed_mask(row_mask, matrix.planes.shape[1])
    rows = np.flatnonzero(allowed)
    if not len(rows):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    row_scores = reference_scores(matrix, query, rows=rows)
    scores = np.full(matrix.planes.shape[1], -np.inf, dtype=np.float64)
    scores[rows] = row_scores
    selected = candidate_order_v1(
        scores, required_kind_mask=required_kind_mask, row_mask=allowed,
        candidate_width=candidate_width, primary_width=primary_width)
    return selected, scores[selected]


def _search_summary(stats):
    if stats is None:
        return None
    return {
        'rounds': stats['rounds'],
        'plane_row_reads': stats['plane_row_reads'],
        'full_plane_row_reads': stats['full_plane_row_reads'],
        'weight_bytes_read': stats['weight_bytes_read'],
        'weight_bytes_full': stats['weight_bytes_full'],
        'scale_bytes_read': stats['scale_bytes_read'],
        'allowed_rows': stats['allowed_rows'],
        'surviving_rows': stats['surviving_rows'],
    }


def progressive_candidate_order(matrix, query, *, required_kind_mask=None,
                                row_mask=None, candidate_width=CANDIDATE_WIDTH,
                                primary_width=PRIMARY_WIDTH,
                                overall_width=OVERALL_WIDTH,
                                plane_schedule=None, eps_rel=1.0e-9):
    """Select from the union of global top-80 and required-kind top-32."""
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    total = matrix.planes.shape[1]
    allowed = _allowed_mask(row_mask, total)
    allowed_rows = int(allowed.sum())
    if not allowed_rows:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0, dtype=np.float64), {
            'overall': None,
            'modality': None,
            'rounds': [],
            'rounds_total': 0,
            'weight_bytes_read': 0,
            'weight_bytes_full': 0,
            'weight_read_fraction': None,
            'codes_sha256': matrix.codes_sha256,
        }

    overall_indices, overall_scores, overall_stats = progressive_topk(
        matrix, query, min(overall_width, allowed_rows), row_mask=allowed,
        plane_schedule=plane_schedule, eps_rel=eps_rel)
    modality_stats = None
    if required_kind_mask is None:
        selected = overall_indices[:candidate_width]
        scores = overall_scores[:candidate_width]
    else:
        kind = _allowed_mask(required_kind_mask, total)
        modality = allowed & kind
        modality_rows = int(modality.sum())
        if modality_rows:
            modality_indices, _, modality_stats = progressive_topk(
                matrix, query, min(primary_width, modality_rows),
                row_mask=modality, plane_schedule=plane_schedule,
                eps_rel=eps_rel)
            union = np.unique(np.concatenate((overall_indices, modality_indices)))
        else:
            union = np.unique(overall_indices)
        union_scores = reference_scores(matrix, query, rows=union)
        local_order = candidate_order_v1(
            union_scores, required_kind_mask=kind[union], row_mask=allowed[union],
            candidate_width=candidate_width, primary_width=primary_width)
        selected = union[local_order]
        scores = union_scores[local_order]

    searches = [overall_stats] + ([modality_stats] if modality_stats else [])
    rounds = []
    for name, search in (('overall', overall_stats), ('modality', modality_stats)):
        if search is not None:
            rounds.extend({'selection': name, **item} for item in search['rounds'])
    weight_read = sum(item['weight_bytes_read'] for item in searches)
    weight_full = sum(item['weight_bytes_full'] for item in searches)
    summary = {
        'overall': _search_summary(overall_stats),
        'modality': _search_summary(modality_stats),
        'rounds': rounds,
        'rounds_total': len(rounds),
        'weight_bytes_read': weight_read,
        'weight_bytes_full': weight_full,
        'weight_read_fraction': weight_read / weight_full if weight_full else None,
        'codes_sha256': matrix.codes_sha256,
    }
    return selected, scores, summary


def overlap_at_k(left, right, k=CANDIDATE_WIDTH):
    if (isinstance(k, bool) or not isinstance(k, int) or k < 1):
        raise ValueError('positive overlap width required')
    left_values = {int(value) for value in np.asarray(left).reshape(-1)[:k]}
    right_values = {int(value) for value in np.asarray(right).reshape(-1)[:k]}
    denominator = min(k, len(left_values), len(right_values))
    count = len(left_values & right_values)
    return count, (count / denominator if denominator else None)


def _map_current_indices(indices, index_map, total):
    if index_map is None:
        return np.asarray(indices, dtype=np.int64)
    mapping = np.asarray(index_map)
    if (mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer)
            or mapping.shape != (total,)):
        raise ValueError('index map must contain one integer per matrix row')
    current = mapping[np.asarray(indices, dtype=np.int64)]
    if np.any(current < 0):
        raise ValueError('progressive selection includes rows outside the current index')
    return current.astype(np.int64, copy=False)


def select_dense_candidates(
        scores, *, mode='off', required_kind_mask=None, row_mask=None,
        candidate_width=CANDIDATE_WIDTH, primary_width=PRIMARY_WIDTH,
        matrix=None, query=None, progressive_required_kind_mask=None,
        progressive_row_mask=None, index_map=None, plane_schedule=None,
        eps_rel=1.0e-9):
    """Select dense candidates while preserving the v1 path when mode is off."""
    if mode not in ('off', 'shadow', 'replace'):
        raise ValueError('mode must be off, shadow, or replace')
    values = np.asarray(scores)
    v1_order = candidate_order_v1(
        values, required_kind_mask=required_kind_mask, row_mask=row_mask,
        candidate_width=candidate_width, primary_width=primary_width)
    if mode == 'off':
        return v1_order, values[v1_order], None
    if matrix is None or query is None:
        raise ValueError('progressive retrieval requires a matrix and query')
    if progressive_row_mask is None:
        progressive_row_mask = row_mask
    if progressive_required_kind_mask is None:
        progressive_required_kind_mask = required_kind_mask
    order, order_scores, search = progressive_candidate_order(
        matrix, query, required_kind_mask=progressive_required_kind_mask,
        row_mask=progressive_row_mask, candidate_width=candidate_width,
        primary_width=primary_width, plane_schedule=plane_schedule,
        eps_rel=eps_rel)
    current_order = _map_current_indices(
        order, index_map, matrix.planes.shape[1])
    overlap_count, overlap_fraction = overlap_at_k(
        v1_order, current_order, candidate_width)
    exactness_assertion = 'not_run_replace'
    if mode == 'shadow':
        full_order, full_scores = quantized_candidate_order(
            matrix, query,
            required_kind_mask=progressive_required_kind_mask,
            row_mask=progressive_row_mask, candidate_width=candidate_width,
            primary_width=primary_width)
        if (not np.array_equal(full_order, order)
                or not np.array_equal(full_scores, order_scores)):
            raise AssertionError(
                'progressive retrieval differs from quantized full selection')
        exactness_assertion = 'passed'
        returned_order = v1_order
        returned_scores = values[v1_order]
    else:
        returned_order = current_order
        returned_scores = order_scores
    stats = {
        'mode': mode,
        'candidate_width': candidate_width,
        'primary_width': primary_width,
        'candidate_count': int(len(returned_order)),
        'float_candidate_count': int(len(v1_order)),
        'overlap_at_48_count': int(overlap_count),
        'overlap_at_48': overlap_fraction,
        'rounds': search['rounds'],
        'rounds_total': int(search['rounds_total']),
        'overall': search['overall'],
        'modality': search['modality'],
        'weight_bytes_read': int(search['weight_bytes_read']),
        'weight_bytes_full': int(search['weight_bytes_full']),
        'weight_read_fraction': search['weight_read_fraction'],
        'codes_sha256': search['codes_sha256'],
        'exactness_assertion': exactness_assertion,
        'quantized_matches_full': (
            True if exactness_assertion == 'passed' else None),
    }
    return returned_order, returned_scores, stats


def disabled_stats():
    return {
        'enabled': False,
        'mode': 'off',
        'demands': [],
        'demand_count': 0,
        'shadow_assertions_passed': None,
        'boundary': 'retrieval progression disabled; v1 float32 path unchanged',
    }


def aggregate_retrieval_stats(mode, demands, *, bits, group_size,
                              codes_sha256):
    """Aggregate per-demand retrieval measurements for a run output."""
    if mode == 'off':
        return disabled_stats()
    if mode not in ('shadow', 'replace'):
        raise ValueError('mode must be off, shadow, or replace')
    items = list(demands)
    rounds_total = sum(int(item.get('rounds_total', 0)) for item in items)
    weight_read = sum(int(item.get('weight_bytes_read', 0)) for item in items)
    weight_full = sum(int(item.get('weight_bytes_full', 0)) for item in items)
    overlaps = [
        float(item['overlap_at_48']) for item in items
        if item.get('overlap_at_48') is not None
    ]
    passed = sum(
        item.get('exactness_assertion') == 'passed' for item in items)
    if mode == 'shadow':
        assertions_passed = (
            passed == len(items) if items else None)
    else:
        assertions_passed = None
    return {
        'enabled': True,
        'mode': mode,
        'bits': bits,
        'group_size': group_size,
        'codes_sha256': codes_sha256,
        'candidate_width': CANDIDATE_WIDTH,
        'primary_width': PRIMARY_WIDTH,
        'overall_width': OVERALL_WIDTH,
        'demands': items,
        'demand_count': len(items),
        'rounds_total': rounds_total,
        'mean_rounds_per_demand': (
            rounds_total / len(items) if items else None),
        'weight_bytes_read': weight_read,
        'weight_bytes_full': weight_full,
        'weight_read_fraction': (
            weight_read / weight_full if weight_full else None),
        'mean_overlap_at_48': (
            sum(overlaps) / len(overlaps) if overlaps else None),
        'exactness_assertions_checked': passed,
        'shadow_assertions_passed': assertions_passed,
        'physical_pim_executed': False,
        'performance_comparable': False,
        'boundary': (
            'Host reference retrieval over plane-major 12-bit codes; '
            'no physical PIM latency or energy is claimed'),
    }
