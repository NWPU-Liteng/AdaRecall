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


@dataclass(frozen=True)
class TorchPlaneMajorMatrix:
    planes: torch.Tensor
    scales: torch.Tensor
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
    return np.ascontiguousarray(codes), np.ascontiguousarray(scales)


def _codes_digest(codes, bits, group_size):
    digest = hashlib.sha256()
    header = np.asarray([bits, group_size, *codes.shape], dtype='<i8')
    digest.update(header.tobytes())
    for begin in range(0, codes.shape[0], 8192):
        chunk = np.ascontiguousarray(codes[begin:begin + 8192], dtype='<i4')
        digest.update(chunk.tobytes())
    return digest.hexdigest()


def _pack_planes(codes, bits):
    rows, cols = codes.shape
    plane_bytes = math.ceil(cols / 8)
    padded_cols = plane_bytes * 8
    planes = np.empty((bits, rows, plane_bytes), dtype=np.uint8)
    mask = (1 << bits) - 1
    for begin in range(0, rows, 8192):
        end = min(begin + 8192, rows)
        unsigned = np.ascontiguousarray(codes[begin:end], dtype=np.int32) & mask
        for plane in range(bits):
            values = ((unsigned >> (bits - 1 - plane)) & 1).astype(np.uint8)
            if padded_cols != cols:
                values = np.pad(values, ((0, 0), (0, padded_cols - cols)))
            planes[plane, begin:end] = np.packbits(
                values, axis=1, bitorder='big')
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


def _pairwise_sum_last(values):
    """Reproduce NumPy's pairwise summation order on the final axis."""
    result = np.asarray(values)
    count = result.shape[-1]
    if count < 8:
        total = np.zeros(result.shape[:-1], dtype=result.dtype)
        for index in range(count):
            total = total + result[..., index]
        return total
    if count <= 128:
        accumulators = [result[..., index] for index in range(8)]
        index = 8
        stop = count - (count % 8)
        while index < stop:
            for lane in range(8):
                accumulators[lane] = (accumulators[lane]
                                      + result[..., index + lane])
            index += 8
        total = ((accumulators[0] + accumulators[1])
                 + (accumulators[2] + accumulators[3]))
        total = total + ((accumulators[4] + accumulators[5])
                         + (accumulators[6] + accumulators[7]))
        for tail in range(index, count):
            total = total + result[..., tail]
        return total
    middle = count // 2
    middle -= middle % 8
    return (_pairwise_sum_last(result[..., :middle])
            + _pairwise_sum_last(result[..., middle:]))


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
    if isinstance(matrix, TorchPlaneMajorMatrix):
        return _reference_scores_torch(matrix, x, rows=rows)
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    x = _validate_activation(matrix, x)
    rows = _row_selection(rows, matrix.planes.shape[1])
    codes = _decode_planes(matrix.planes, matrix.bits, matrix.cols)[rows]
    products = codes.astype(np.float64).reshape(
        len(rows), matrix.cols // matrix.group_size, matrix.group_size)
    activation = x.reshape(1, matrix.cols // matrix.group_size, matrix.group_size)
    grouped_scores = _pairwise_sum_last(products * activation)
    return _pairwise_sum_last(grouped_scores * matrix.scales[rows])


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
    grouped_scores = _pairwise_sum_last(products * activation)
    partial = _pairwise_sum_last(grouped_scores * matrix.scales[rows])
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
    positive_groups = _pairwise_sum_last(np.maximum(grouped, 0))
    negative_groups = _pairwise_sum_last(np.minimum(grouped, 0))
    positive = _pairwise_sum_last(
        matrix.scales * positive_groups.reshape(1, -1))
    negative = _pairwise_sum_last(
        matrix.scales * negative_groups.reshape(1, -1))

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


def to_torch_matrix(matrix, *, device=None):
    """Move a validated reference matrix to a torch device without repacking."""
    if not isinstance(matrix, PlaneMajorMatrix):
        raise TypeError('PlaneMajorMatrix required')
    planes = torch.from_numpy(np.ascontiguousarray(matrix.planes))
    scales = torch.from_numpy(np.ascontiguousarray(matrix.scales))
    if device is not None:
        planes = planes.to(device)
        scales = scales.to(device)
    result = TorchPlaneMajorMatrix(
        planes=planes, scales=scales, bits=matrix.bits,
        group_size=matrix.group_size, cols=matrix.cols,
        codes_sha256=matrix.codes_sha256)
    _validate_torch_matrix(result)
    return result


def _validate_torch_matrix(matrix):
    if not isinstance(matrix, TorchPlaneMajorMatrix):
        raise TypeError('TorchPlaneMajorMatrix required')
    if (not torch.is_tensor(matrix.planes) or not torch.is_tensor(matrix.scales)
            or matrix.planes.dtype != torch.uint8
            or matrix.scales.dtype != torch.float64):
        raise ValueError('torch uint8 planes and float64 scales required')
    if (matrix.planes.ndim != 3 or matrix.planes.shape[0] != matrix.bits
            or matrix.planes.shape[2] != math.ceil(matrix.cols / 8)
            or matrix.planes.device != matrix.scales.device):
        raise ValueError('torch plane shape or device does not match the matrix')
    expected_scales = (matrix.planes.shape[1], matrix.cols // matrix.group_size)
    if matrix.scales.shape != expected_scales:
        raise ValueError('torch scales must have shape [rows, columns/group_size]')
    if not torch.isfinite(matrix.scales).all().item() or not (matrix.scales > 0).all().item():
        raise ValueError('finite positive torch group scales required')
    return matrix


def _validate_activation_torch(matrix, x):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64, device=matrix.planes.device)
    else:
        x = x.to(device=matrix.planes.device, dtype=torch.float64)
    if x.shape != (matrix.cols,) or not torch.isfinite(x).all().item():
        raise ValueError('one finite float64 activation row matching columns required')
    return x


def _row_selection_torch(rows, total, device):
    if rows is None:
        return torch.arange(total, dtype=torch.int64, device=device)
    if torch.is_tensor(rows):
        values = rows.to(device=device)
    else:
        values = torch.as_tensor(rows, device=device)
    if values.dtype == torch.bool:
        if values.shape != (total,):
            raise ValueError('row mask must match rows')
        return torch.nonzero(values, as_tuple=False).flatten().to(torch.int64)
    if values.ndim != 1 or values.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError('rows must be integer indices or a boolean mask')
    indices = values.to(torch.int64)
    if (bool((indices < 0).any().item())
            or bool((indices >= total).any().item())
            or len(torch.unique(indices)) != len(indices)):
        raise ValueError('unique in-range row indices required')
    return indices


def _pairwise_sum_last_torch(values):
    """Torch port of NumPy's eight-accumulator pairwise summation."""
    result = values
    count = result.shape[-1]
    if count < 8:
        total = torch.zeros(
            result.shape[:-1], dtype=result.dtype, device=result.device)
        for index in range(count):
            total = total + result[..., index]
        return total
    if count <= 128:
        accumulators = [result[..., index] for index in range(8)]
        index = 8
        stop = count - (count % 8)
        while index < stop:
            for lane in range(8):
                accumulators[lane] = (accumulators[lane]
                                      + result[..., index + lane])
            index += 8
        total = ((accumulators[0] + accumulators[1])
                 + (accumulators[2] + accumulators[3]))
        total = total + ((accumulators[4] + accumulators[5])
                         + (accumulators[6] + accumulators[7]))
        for tail in range(index, count):
            total = total + result[..., tail]
        return total
    middle = count // 2
    middle -= middle % 8
    return (_pairwise_sum_last_torch(result[..., :middle])
            + _pairwise_sum_last_torch(result[..., middle:]))


def _decode_planes_torch(planes, bits, cols):
    if (planes.ndim != 3 or planes.shape[0] != bits
            or planes.shape[2] != math.ceil(cols / 8)):
        raise ValueError('plane shape does not match bits and columns')
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=planes.device)
    bit_values = ((planes.unsqueeze(-1) >> shifts) & 1).to(torch.int32)
    bit_values = bit_values.reshape(bits, planes.shape[1], -1)[:, :, :cols]
    codes = bit_values[0] * (-(1 << (bits - 1)))
    for plane in range(1, bits):
        codes = codes + (bit_values[plane] << (bits - 1 - plane))
    return codes


def _partial_codes_torch(matrix, rows, stop_plane):
    planes = matrix.planes[:stop_plane, rows]
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=planes.device)
    bit_values = ((planes.unsqueeze(-1) >> shifts) & 1).to(torch.int32)
    bit_values = bit_values.reshape(
        stop_plane, len(rows), -1)[:, :, :matrix.cols]
    codes = bit_values[0] * (-(1 << (matrix.bits - 1)))
    for plane in range(1, stop_plane):
        codes = codes + (bit_values[plane] << (matrix.bits - 1 - plane))
    return codes


def _reference_scores_torch(matrix, x, rows=None, *, chunk_rows=4096):
    _validate_torch_matrix(matrix)
    x = _validate_activation_torch(matrix, x)
    rows = _row_selection_torch(rows, matrix.planes.shape[1], matrix.planes.device)
    groups = matrix.cols // matrix.group_size
    activation = x.reshape(1, groups, matrix.group_size)
    output = torch.empty(len(rows), dtype=torch.float64, device=matrix.planes.device)
    for begin in range(0, len(rows), chunk_rows):
        end = min(begin + chunk_rows, len(rows))
        selected = rows[begin:end]
        codes = _decode_planes_torch(
            matrix.planes[:, selected], matrix.bits, matrix.cols)
        products = codes.to(torch.float64).reshape(
            len(selected), groups, matrix.group_size) * activation
        grouped_scores = _pairwise_sum_last_torch(products)
        output[begin:end] = _pairwise_sum_last_torch(
            grouped_scores * matrix.scales[selected])
    return output


def _scores_and_bounds_torch(matrix, x, rows, stop_plane, positive, negative,
                             *, chunk_rows=4096):
    groups = matrix.cols // matrix.group_size
    activation = x.reshape(1, groups, matrix.group_size)
    partial = torch.empty(len(rows), dtype=torch.float64, device=x.device)
    for begin in range(0, len(rows), chunk_rows):
        end = min(begin + chunk_rows, len(rows))
        selected = rows[begin:end]
        codes = _partial_codes_torch(
            matrix, selected, stop_plane).to(torch.float64)
        products = codes.reshape(len(selected), groups, matrix.group_size) * activation
        grouped_scores = _pairwise_sum_last_torch(products)
        partial[begin:end] = _pairwise_sum_last_torch(
            grouped_scores * matrix.scales[selected])
    residual = (1 << (matrix.bits - stop_plane)) - 1
    lower = partial + residual * negative[rows]
    upper = partial + residual * positive[rows]
    return partial, lower, upper


def _torch_topk(exact, alive, k):
    by_index = torch.argsort(alive, stable=True)
    ordered_scores = exact[by_index]
    score_order = torch.argsort(
        ordered_scores, descending=True, stable=True)[:min(k, len(alive))]
    selected = by_index[score_order]
    return alive[selected], exact[selected]


def progressive_topk_torch(matrix, x, k, row_mask=None, plane_schedule=None,
                           eps_rel=1e-9, *, chunk_rows=4096):
    """GPU/CPU torch port bitwise-equivalent to :func:`progressive_topk`."""
    _validate_torch_matrix(matrix)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('positive integer k required')
    if (isinstance(eps_rel, bool) or not isinstance(eps_rel, (int, float))
            or not math.isfinite(float(eps_rel)) or eps_rel < 0):
        raise ValueError('nonnegative finite eps_rel required')
    x = _validate_activation_torch(matrix, x)
    schedule = _validate_schedule(plane_schedule, matrix.bits)
    allowed = _row_selection_torch(
        row_mask, matrix.planes.shape[1], matrix.planes.device)
    if not len(allowed):
        raise ValueError('at least one allowed row required')

    groups = matrix.cols // matrix.group_size
    grouped = x.reshape(groups, matrix.group_size)
    positive_groups = _pairwise_sum_last_torch(torch.clamp_min(grouped, 0))
    negative_groups = _pairwise_sum_last_torch(torch.clamp_max(grouped, 0))
    positive = _pairwise_sum_last_torch(
        matrix.scales * positive_groups.reshape(1, -1))
    negative = _pairwise_sum_last_torch(
        matrix.scales * negative_groups.reshape(1, -1))

    alive = allowed
    stop_plane = 0
    rounds = []
    plane_row_reads = 0
    plane_bytes = math.ceil(matrix.cols / 8)
    for planes_in_round in schedule:
        before = len(alive)
        stop_plane += planes_in_round
        plane_row_reads += before * planes_in_round
        _, lower, upper = _scores_and_bounds_torch(
            matrix, x, alive, stop_plane, positive, negative,
            chunk_rows=chunk_rows)
        if before > k:
            tau = torch.kthvalue(lower, before - k + 1).values
            epsilon = float(eps_rel) * max(1.0, float(upper.abs().max().item()))
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

    exact = _reference_scores_torch(
        matrix, x, rows=alive, chunk_rows=chunk_rows)
    indices, scores = _torch_topk(exact, alive, k)
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
