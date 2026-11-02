"""Reference plane-major 12-bit progressive top-k over quantized rows."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from .functional_decode import quantize_signed_groups


@dataclass(frozen=True)
class PlaneMajorMatrix:
    planes: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int
    cols: int
    codes_sha256: str


def _validate_codes(codes, scales, bits, group_size):
    codes = np.asarray(codes)
    scales = np.asarray(scales, dtype=np.float64)
    if codes.ndim != 2 or codes.shape[0] < 1 or codes.shape[1] < 1:
        raise ValueError('codes must be a nonempty two-dimensional matrix')
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 31:
        raise ValueError('bits must be an integer in [1, 31]')
    if (isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size < 1 or codes.shape[1] % group_size):
        raise ValueError('columns must align to a positive group size')
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('codes must use an integer dtype')
    if codes.shape[0] != scales.shape[0] or scales.shape[1] != codes.shape[1] // group_size:
        raise ValueError('scales must have shape [rows, columns/group_size]')
    if not np.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError('finite positive group scales required')
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if np.any(codes < lower) or np.any(codes > upper):
        raise ValueError(f'codes must fit signed {bits}-bit two\'s complement')
    return np.ascontiguousarray(codes, dtype=np.int32), np.ascontiguousarray(scales)


def _codes_digest(codes, bits, group_size):
    digest = hashlib.sha256()
    header = np.asarray([bits, group_size, *codes.shape], dtype='<i8')
    digest.update(header.tobytes())
    digest.update(np.ascontiguousarray(codes, dtype='<i4').tobytes())
    return digest.hexdigest()


def _pack_planes(codes, bits):
    rows, cols = codes.shape
    plane_bytes = math.ceil(cols / 8)
    padded_cols = plane_bytes * 8
    unsigned = np.asarray(codes, dtype=np.int32) & ((1 << bits) - 1)
    planes = np.empty((bits, rows, plane_bytes), dtype=np.uint8)
    for plane in range(bits):
        values = ((unsigned >> (bits - 1 - plane)) & 1).astype(np.uint8)
        if padded_cols != cols:
            values = np.pad(values, ((0, 0), (0, padded_cols - cols)))
        planes[plane] = np.packbits(values, axis=1, bitorder='big')
    return planes


def from_codes(codes, scales, bits, group_size):
    codes, scales = _validate_codes(codes, scales, bits, group_size)
    return PlaneMajorMatrix(
        planes=_pack_planes(codes, bits), scales=scales, bits=bits,
        group_size=group_size, cols=codes.shape[1],
        codes_sha256=_codes_digest(codes, bits, group_size))


def quantize_plane_major(w, bits=12, group_size=128):
    value = np.asarray(w)
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError('a nonempty two-dimensional weight matrix is required')
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError('finite floating-point weights required')
    floating = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    codes, scales, _ = quantize_signed_groups(
        floating, group_size=group_size, weight_bits=bits)
    return from_codes(codes.numpy(), scales.numpy(), bits, group_size)


def _decode_planes(planes, bits, cols):
    if planes.ndim != 3 or planes.shape[0] != bits or planes.shape[2] != math.ceil(cols / 8):
        raise ValueError('plane shape does not match bits and columns')
    bits_array = np.unpackbits(planes, axis=2, bitorder='big')[:, :, :cols]
    codes = bits_array[0].astype(np.int32) * -(1 << (bits - 1))
    for plane in range(1, bits):
        codes += bits_array[plane].astype(np.int32) << (bits - 1 - plane)
    return codes.astype(np.int32, copy=False)


def decode_codes(matrix):
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    return _decode_planes(matrix.planes, matrix.bits, matrix.cols)


def _validate_activation(matrix, x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (matrix.cols,) or not np.isfinite(x).all():
        raise ValueError('one finite float64 activation row matching columns required')
    return x


def _row_selection(rows, total):
    if rows is None:
        return np.arange(total, dtype=np.int64)
    values = np.asarray(rows)
    if values.dtype == np.bool_:
        if values.shape != (total,):
            raise ValueError('row mask must match rows')
        return np.flatnonzero(values).astype(np.int64, copy=False)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('rows must be integer indices or a boolean mask')
    indices = values.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= total) or len(np.unique(indices)) != len(indices):
        raise ValueError('unique in-range row indices required')
    return indices


def reference_scores(matrix, x, rows=None):
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    x = _validate_activation(matrix, x)
    rows = _row_selection(rows, matrix.planes.shape[1])
    codes = _decode_planes(matrix.planes, matrix.bits, matrix.cols)[rows]
    products = codes.astype(np.float64).reshape(
        len(rows), matrix.cols // matrix.group_size, matrix.group_size)
    activation = x.reshape(1, matrix.cols // matrix.group_size, matrix.group_size)
    grouped_scores = (products * activation).sum(axis=2)
    return (grouped_scores * matrix.scales[rows]).sum(axis=1)


def _partial_codes(matrix, rows, stop_plane):
    planes = matrix.planes[:stop_plane, rows]
    bit_values = np.unpackbits(planes, axis=2, bitorder='big')[:, :, :matrix.cols]
    codes = bit_values[0].astype(np.int32) * -(1 << (matrix.bits - 1))
    for plane in range(1, stop_plane):
        codes += bit_values[plane].astype(np.int32) << (matrix.bits - 1 - plane)
    return codes


def _scores_and_bounds(matrix, x, rows, stop_plane, positive, negative):
    codes = _partial_codes(matrix, rows, stop_plane).astype(np.float64)
    groups = matrix.cols // matrix.group_size
    products = codes.reshape(len(rows), groups, matrix.group_size)
    activation = x.reshape(1, groups, matrix.group_size)
    grouped_scores = (products * activation).sum(axis=2)
    partial = (grouped_scores * matrix.scales[rows]).sum(axis=1)
    residual = (1 << (matrix.bits - stop_plane)) - 1
    lower = partial + residual * negative[rows]
    upper = partial + residual * positive[rows]
    return partial, lower, upper


def _validate_schedule(plane_schedule, bits):
    schedule = list(plane_schedule) if plane_schedule is not None else [1] * bits
    if (not schedule or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                            for value in schedule) or sum(schedule) != bits):
        raise ValueError('positive plane schedule must sum to bits')
    return schedule


def progressive_topk(matrix, x, k, row_mask=None, plane_schedule=None, eps_rel=1e-9):
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('positive integer k required')
    if isinstance(eps_rel, bool) or not isinstance(eps_rel, (int, float)) \
            or not math.isfinite(float(eps_rel)) or eps_rel < 0:
        raise ValueError('nonnegative finite eps_rel required')
    x = _validate_activation(matrix, x)
    schedule = _validate_schedule(plane_schedule, matrix.bits)
    allowed = _row_selection(row_mask, matrix.planes.shape[1])
    if not len(allowed):
        raise ValueError('at least one allowed row required')

    groups = matrix.cols // matrix.group_size
    grouped = x.reshape(groups, matrix.group_size)
    positive_groups = np.maximum(grouped, 0).sum(axis=1)
    negative_groups = np.minimum(grouped, 0).sum(axis=1)
    positive = matrix.scales.dot(positive_groups)
    negative = matrix.scales.dot(negative_groups)

    alive = allowed.copy()
    stop_plane = 0
    rounds = []
    plane_row_reads = 0
    plane_bytes = math.ceil(matrix.cols / 8)
    for planes_in_round in schedule:
        before = len(alive)
        stop_plane += planes_in_round
        plane_row_reads += before * planes_in_round
        partial, lower, upper = _scores_and_bounds(
            matrix, x, alive, stop_plane, positive, negative)
        if before > k:
            tau = np.partition(lower, before - k)[before - k]
            epsilon = float(eps_rel) * max(1.0, float(np.max(np.abs(upper))))
            keep = upper >= tau - epsilon
            alive = alive[keep]
        rounds.append({
            'planes_in_round': planes_in_round,
            'planes_read_total': stop_plane,
            'rows_alive_before': before,
            'rows_alive_after': len(alive),
        })
        if not len(alive):
            raise AssertionError('pruning removed every row')

    exact = reference_scores(matrix, x, rows=alive)
    order = np.lexsort((alive, -exact))[:min(k, len(alive))]
    indices = alive[order].astype(np.int64, copy=False)
    scores = exact[order]
    full_plane_row_reads = len(allowed) * matrix.bits
    scale_bytes_read = sum(
        item['rows_alive_before'] * groups * 4 for item in rounds)
    stats = {
        'rounds': rounds,
        'plane_row_reads': plane_row_reads,
        'full_plane_row_reads': full_plane_row_reads,
        'weight_bytes_read': plane_row_reads * plane_bytes,
        'weight_bytes_full': full_plane_row_reads * plane_bytes,
        'scale_bytes_read': scale_bytes_read,
        'allowed_rows': len(allowed),
        'surviving_rows': len(alive),
        'planes_total': matrix.bits,
        'codes_sha256': matrix.codes_sha256,
    }
    return indices, scores, stats
