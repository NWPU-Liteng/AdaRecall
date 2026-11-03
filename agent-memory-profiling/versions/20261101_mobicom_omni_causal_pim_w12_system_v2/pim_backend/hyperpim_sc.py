"""Bank-resident decoder accounting for HyperPIM-SC."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .structured_control import StructuredControlConfig


@dataclass(frozen=True)
class DecodeCostConfig:
    host_linear_ms_per_token: float
    pim_linear_ms_per_token: float
    gpu_nonlinear_ms_per_token: float
    grammar_scan_ms_per_token: float
    dispatch_ms_per_token: float
    safety_factor: float = 1.0

    def __post_init__(self) -> None:
        values = (self.host_linear_ms_per_token, self.pim_linear_ms_per_token,
                  self.gpu_nonlinear_ms_per_token, self.grammar_scan_ms_per_token,
                  self.dispatch_ms_per_token, self.safety_factor)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("finite nonnegative decode costs required")
        if self.pim_linear_ms_per_token + self.gpu_nonlinear_ms_per_token <= 0:
            raise ValueError("positive PIM or GPU decode work required")
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1:
            raise ValueError("safety factor must be finite and at least one")

    def target_ms(self, *, structured: bool) -> float:
        grammar = self.grammar_scan_ms_per_token if structured else 0.0
        return self.safety_factor * (
            self.pim_linear_ms_per_token + self.gpu_nonlinear_ms_per_token
            + grammar + self.dispatch_ms_per_token)

    def weight_stream_speedup(self) -> float:
        if self.pim_linear_ms_per_token <= 0:
            raise ValueError("positive PIM linear cost required")
        return self.host_linear_ms_per_token / self.pim_linear_ms_per_token


@dataclass(frozen=True)
class W8A16CostConfig:
    """Conservative bank-local mixed-precision execution contract.

    The weight tile remains INT8 in every PIM bank.  BF16 activations are
    broadcast per matrix, converted weights are consumed by a mixed-precision
    MAC, and grouped scales are fetched from the same bank.  Conversion,
    scale application and a possible BF16 MAC slowdown are charged explicitly
    rather than hidden in the INT8 GEMV calibration.
    """

    group_size: int = 128
    scale_bytes: int = 4
    conversion_elements_per_bank_cycle: int = 16
    group_apply_ops_per_bank_cycle: int = 16
    matrix_command_cycles: int = 64
    mac_slowdown: float = 1.0
    activation_bytes_per_element: int = 2
    broadcast_bandwidth_bytes_per_second: float = 90_069_343_391.2629
    residual_bits: int = 0
    residual_scale_bytes: int = 4
    residual_mac_relative_cost: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.group_size, self.scale_bytes,
            self.conversion_elements_per_bank_cycle,
            self.group_apply_ops_per_bank_cycle, self.matrix_command_cycles,
            self.activation_bytes_per_element, self.residual_scale_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in integer_fields):
            raise ValueError("positive integer W8A16 contract fields required")
        if self.residual_bits not in (0, 2, 4):
            raise ValueError("residual bits must be zero, two or four")
        if (not math.isfinite(self.mac_slowdown) or self.mac_slowdown < 1
                or not math.isfinite(self.broadcast_bandwidth_bytes_per_second)
                or self.broadcast_bandwidth_bytes_per_second <= 0
                or not math.isfinite(self.residual_mac_relative_cost)
                or self.residual_mac_relative_cost < 1):
            raise ValueError("finite MAC slowdown >= 1 and positive broadcast bandwidth required")

    def scale_bytes_per_row(self, in_features: int) -> int:
        if isinstance(in_features, bool) or not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("positive integer input width required")
        return math.ceil(in_features / self.group_size) * self.scale_bytes


def bank_resident_decoder_plan(matrices, config: StructuredControlConfig | None = None,
                               *, scale_bytes_per_row: int = 4,
                               group_size: int | None = None,
                               scale_bytes_per_group: int = 4,
                               residual_bits: int = 0,
                               residual_scale_bytes: int = 4) -> dict:
    """Place one W8 copy by output row across the configured bank topology."""
    config = config or StructuredControlConfig()
    if isinstance(scale_bytes_per_row, bool) or scale_bytes_per_row < 0:
        raise ValueError("nonnegative integer scale bytes per row required")
    if residual_bits not in (0, 2, 4):
        raise ValueError("residual bits must be zero, two or four")
    if isinstance(residual_scale_bytes, bool) or residual_scale_bytes < 0:
        raise ValueError("nonnegative integer residual scale bytes required")
    if group_size is not None:
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError("positive integer group size required")
        if (isinstance(scale_bytes_per_group, bool)
                or not isinstance(scale_bytes_per_group, int) or scale_bytes_per_group <= 0):
            raise ValueError("positive integer scale bytes per group required")
    bank_weight_bytes = [0] * config.bank_count
    bank_scale_bytes = [0] * config.bank_count
    bank_residual_bytes = [0] * config.bank_count
    records = []
    logical_weight_bytes = 0
    logical_scale_bytes = 0
    logical_residual_bytes = 0
    logical_scale_groups = 0
    for matrix in matrices:
        name = str(matrix["name"])
        outputs = int(matrix["out_features"])
        inputs = int(matrix["in_features"])
        if not name or outputs <= 0 or inputs <= 0:
            raise ValueError("named positive matrix dimensions required")
        groups_per_row = (math.ceil(inputs / group_size) if group_size is not None else None)
        matrix_scale_bytes_per_row = (groups_per_row * scale_bytes_per_group
                                      if groups_per_row is not None else scale_bytes_per_row)
        residual_bytes_per_row = math.ceil(inputs * residual_bits / 8)
        matrix_residual_scale_bytes = (
            outputs * groups_per_row * residual_scale_bytes
            if residual_bits and groups_per_row is not None else 0)
        bank_residual_per_row = residual_bytes_per_row + (
            matrix_residual_scale_bytes // outputs if outputs else 0)
        rows_per_bank = math.ceil(outputs / config.bank_count)
        for bank in range(min(outputs, config.bank_count)):
            rows = min(rows_per_bank, outputs - bank * rows_per_bank)
            if rows <= 0:
                continue
            bank_weight_bytes[bank] += rows * inputs
            bank_scale_bytes[bank] += rows * matrix_scale_bytes_per_row
            bank_residual_bytes[bank] += rows * bank_residual_per_row
        matrix_bytes = outputs * inputs
        matrix_scale_bytes = outputs * matrix_scale_bytes_per_row
        matrix_residual_bytes = outputs * residual_bytes_per_row
        matrix_scale_groups = outputs * groups_per_row if groups_per_row is not None else 0
        logical_weight_bytes += matrix_bytes
        logical_scale_bytes += matrix_scale_bytes
        logical_residual_bytes += matrix_residual_bytes + matrix_residual_scale_bytes
        logical_scale_groups += matrix_scale_groups
        records.append({
            "name": name, "out_features": outputs, "in_features": inputs,
            "rows_per_bank": rows_per_bank,
            "active_banks": min(outputs, config.bank_count),
            "logical_weight_bytes": matrix_bytes,
            "logical_scale_bytes": matrix_scale_bytes,
            "scale_groups": matrix_scale_groups,
            "residual_bytes": matrix_residual_bytes,
            "residual_scale_bytes": matrix_residual_scale_bytes,
        })
    total_bank_bytes = [weight + scale + residual for weight, scale, residual in
                        zip(bank_weight_bytes, bank_scale_bytes, bank_residual_bytes)]
    padded_weight_bytes = sum(bank_weight_bytes)
    padded_scale_bytes = sum(bank_scale_bytes)
    padded_residual_bytes = sum(bank_residual_bytes)
    return {
        "matrices": records,
        "matrix_count": len(records),
        "banks": config.bank_count,
        "logical_weight_bytes": logical_weight_bytes,
        "logical_scale_bytes": logical_scale_bytes,
        "logical_scale_groups": logical_scale_groups,
        "logical_residual_bytes": logical_residual_bytes,
        "group_size": group_size,
        "scale_bytes_per_group": scale_bytes_per_group if group_size is not None else None,
        "bank_padded_weight_bytes": padded_weight_bytes,
        "bank_padded_scale_bytes": padded_scale_bytes,
        "bank_padded_residual_bytes": padded_residual_bytes,
        "bank_padded_total_bytes": padded_weight_bytes + padded_scale_bytes + padded_residual_bytes,
        "bank_min_bytes": min(total_bank_bytes),
        "bank_max_bytes": max(total_bank_bytes),
        "bank_mean_bytes": sum(total_bank_bytes) / len(total_bank_bytes),
        "bank_imbalance": (max(total_bank_bytes) / (sum(total_bank_bytes) / len(total_bank_bytes))),
        "weight_copies": 1,
        "capacity_bytes": config.capacity_bytes,
        "feasible": (padded_weight_bytes + padded_scale_bytes
                     + padded_residual_bytes <= config.capacity_bytes),
        "placement_boundary": (
            "static row-major bank placement and capacity accounting; no refresh, repair, "
            "metadata duplication or runtime migration is included"),
    }


def packed_bit_sliced_decoder_plan(matrices, *,
                                   weight_bits=12,
                                   group_size=128,
                                   scale_bytes=4,
                                   row_bytes=2048,
                                   capacity_bytes=8 * 1024**3) -> dict:
    """Account a row-packed uniform bit-sliced bank plane.

    A logical weight row is packed contiguously at ``weight_bits`` per element
    and then rounded up to the physical DRAM row.  Unlike a residual plane,
    there is exactly one weight plane and one group-scale plane.  This model is
    capacity-only; execution timing is modeled separately.
    """
    if weight_bits not in (8, 10, 12):
        raise ValueError("weight bits must be eight, ten or twelve")
    integer_fields = (group_size, scale_bytes, row_bytes, capacity_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in integer_fields):
        raise ValueError("positive integer packing and capacity fields required")
    if not isinstance(matrices, (list, tuple)) or not matrices:
        raise ValueError("nonempty matrix inventory required")

    records = []
    padded_weight_bytes = 0
    logical_weight_bytes = 0
    scale_bytes_total = 0
    for matrix in matrices:
        name = str(matrix.get("name") or matrix.get("matrix_id") or "")
        inputs = int(matrix.get("in_features") or 0)
        outputs = int(matrix.get("out_features") or 0)
        if not name or inputs <= 0 or outputs <= 0:
            raise ValueError("named positive matrix dimensions required")
        logical_row_bytes = math.ceil(inputs * weight_bits / 8)
        physical_row_bytes = math.ceil(logical_row_bytes / row_bytes) * row_bytes
        logical_bytes = outputs * logical_row_bytes
        packed_bytes = outputs * physical_row_bytes
        groups = outputs * math.ceil(inputs / group_size)
        matrix_scale_bytes = groups * scale_bytes
        padded_weight_bytes += packed_bytes
        logical_weight_bytes += logical_bytes
        scale_bytes_total += matrix_scale_bytes
        records.append({
            "name": name,
            "out_features": outputs,
            "in_features": inputs,
            "logical_row_bytes": logical_row_bytes,
            "physical_row_bytes": physical_row_bytes,
            "physical_rows_per_output_row": physical_row_bytes // row_bytes,
            "logical_weight_bytes": logical_bytes,
            "packed_weight_bytes": packed_bytes,
            "scale_groups": groups,
            "scale_bytes": matrix_scale_bytes,
        })
    total = padded_weight_bytes + scale_bytes_total
    return {
        "layout": "uniform bit-sliced, row-packed, single weight plane",
        "weight_bits": weight_bits,
        "group_size": group_size,
        "scale_bytes_per_group": scale_bytes,
        "row_bytes": row_bytes,
        "matrices": records,
        "matrix_count": len(records),
        "logical_weight_bytes": logical_weight_bytes,
        "packed_weight_bytes": padded_weight_bytes,
        "scale_bytes": scale_bytes_total,
        "total_bytes": total,
        "row_packing_overhead": padded_weight_bytes / logical_weight_bytes,
        "weight_copies": 1,
        "capacity_bytes": capacity_bytes,
        "headroom_bytes": capacity_bytes - total,
        "feasible": total <= capacity_bytes,
        "capacity_boundary": (
            "capacity model only: one bit-sliced weight plane and one group-scale plane; "
            "does not include token table, grammar state, scratch, refresh metadata, "
            "repair, or runtime migration"),
    }


def w8a16_mixed_precision_overhead(matrices,
                                   pim_linear_ms_per_token: float,
                                   config: StructuredControlConfig | None = None,
                                   cost: W8A16CostConfig | None = None) -> dict:
    """Charge bank conversion, scale handling, broadcast and MAC slowdown.

    Every matrix is processed by all active banks in parallel.  Conversion and
    grouped-scale arithmetic consume the same bank-local lane budget as the
    MAC, so their cycles are additive rather than overlapped.  The activation
    broadcast uses the measured Host-to-bank effective bandwidth, not the
    higher bank-internal weight bandwidth.
    """
    config = config or StructuredControlConfig()
    cost = cost or W8A16CostConfig()
    if (not math.isfinite(pim_linear_ms_per_token) or pim_linear_ms_per_token <= 0):
        raise ValueError("positive finite INT8 PIM linear time required")
    conversion_cycles = 0
    residual_conversion_cycles = 0
    scale_read_cycles = 0
    residual_scale_read_cycles = 0
    scale_apply_cycles = 0
    residual_scale_apply_cycles = 0
    command_cycles = 0
    broadcast_bytes = 0
    weight_elements = 0
    group_count = 0
    records = []
    for matrix in matrices:
        name = str(matrix["name"])
        outputs = int(matrix["out_features"])
        inputs = int(matrix["in_features"])
        if not name or outputs <= 0 or inputs <= 0:
            raise ValueError("named positive matrix dimensions required")
        rows_per_bank = math.ceil(outputs / config.bank_count)
        elements_per_bank = rows_per_bank * inputs
        groups_per_bank = rows_per_bank * math.ceil(inputs / cost.group_size)
        matrix_conversion = math.ceil(
            elements_per_bank / cost.conversion_elements_per_bank_cycle)
        matrix_scale_read = math.ceil(
            groups_per_bank * cost.scale_bytes / config.bank_internal_bytes_per_cycle)
        matrix_scale_apply = math.ceil(
            groups_per_bank / cost.group_apply_ops_per_bank_cycle)
        matrix_residual_conversion = math.ceil(
            elements_per_bank * cost.residual_bits / 8
            / cost.conversion_elements_per_bank_cycle) if cost.residual_bits else 0
        matrix_residual_scale_read = math.ceil(
            groups_per_bank * cost.residual_scale_bytes
            / config.bank_internal_bytes_per_cycle) if cost.residual_bits else 0
        matrix_residual_scale_apply = math.ceil(
            groups_per_bank / cost.group_apply_ops_per_bank_cycle
        ) if cost.residual_bits else 0
        conversion_cycles += matrix_conversion
        residual_conversion_cycles += matrix_residual_conversion
        scale_read_cycles += matrix_scale_read
        residual_scale_read_cycles += matrix_residual_scale_read
        scale_apply_cycles += matrix_scale_apply
        residual_scale_apply_cycles += matrix_residual_scale_apply
        command_cycles += cost.matrix_command_cycles
        broadcast_bytes += inputs * cost.activation_bytes_per_element
        weight_elements += outputs * inputs
        group_count += outputs * math.ceil(inputs / cost.group_size)
        records.append({
            "name": name,
            "elements_per_bank": elements_per_bank,
            "scale_groups_per_bank": groups_per_bank,
            "conversion_cycles": matrix_conversion,
            "scale_read_cycles": matrix_scale_read,
            "scale_apply_cycles": matrix_scale_apply,
            "residual_bits": cost.residual_bits,
            "residual_conversion_cycles": matrix_residual_conversion,
            "residual_scale_read_cycles": matrix_residual_scale_read,
            "residual_scale_apply_cycles": matrix_residual_scale_apply,
        })
    arithmetic_cycles = (conversion_cycles + residual_conversion_cycles
                         + scale_read_cycles + residual_scale_read_cycles
                         + scale_apply_cycles + residual_scale_apply_cycles
                         + command_cycles)
    arithmetic_ms = 1000 * arithmetic_cycles / config.clock_hz
    broadcast_ms = 1000 * broadcast_bytes / cost.broadcast_bandwidth_bytes_per_second
    mac_slowdown_ms = pim_linear_ms_per_token * (cost.mac_slowdown - 1.0)
    residual_mac_ms = (pim_linear_ms_per_token * cost.residual_mac_relative_cost
                       if cost.residual_bits else 0.0)
    total_overhead_ms = arithmetic_ms + broadcast_ms + mac_slowdown_ms
    return {
        "contract": {
            "weights": "int8_bank_resident_single_copy",
            "activations": "bf16_streamed_per_matrix",
            "group_size": cost.group_size,
            "scale_bytes": cost.scale_bytes,
            "conversion_elements_per_bank_cycle": cost.conversion_elements_per_bank_cycle,
            "group_apply_ops_per_bank_cycle": cost.group_apply_ops_per_bank_cycle,
            "matrix_command_cycles": cost.matrix_command_cycles,
            "mac_slowdown": cost.mac_slowdown,
            "activation_bytes_per_element": cost.activation_bytes_per_element,
            "broadcast_bandwidth_bytes_per_second": cost.broadcast_bandwidth_bytes_per_second,
            "residual_bits": cost.residual_bits,
            "residual_scale_bytes": cost.residual_scale_bytes,
            "residual_mac_relative_cost": cost.residual_mac_relative_cost,
        },
        "matrices": records,
        "weight_elements": weight_elements,
        "scale_groups": group_count,
        "conversion_cycles": conversion_cycles,
        "residual_conversion_cycles": residual_conversion_cycles,
        "scale_read_cycles": scale_read_cycles,
        "residual_scale_read_cycles": residual_scale_read_cycles,
        "scale_apply_cycles": scale_apply_cycles,
        "residual_scale_apply_cycles": residual_scale_apply_cycles,
        "matrix_command_cycles": command_cycles,
        "arithmetic_cycles": arithmetic_cycles,
        "arithmetic_ms_per_token": arithmetic_ms,
        "activation_broadcast_bytes_per_token": broadcast_bytes,
        "activation_broadcast_ms_per_token": broadcast_ms,
        "mac_slowdown_ms_per_token": mac_slowdown_ms,
        "residual_mac_ms_per_token": residual_mac_ms,
        "total_overhead_ms_per_token": total_overhead_ms,
        "pim_linear_ms_per_token": pim_linear_ms_per_token,
        "pim_linear_with_overhead_ms_per_token": (
            pim_linear_ms_per_token + residual_mac_ms + total_overhead_ms),
        "overhead_fraction_of_int8_linear": (
            (residual_mac_ms + total_overhead_ms) / pim_linear_ms_per_token),
        "boundary": ("analytical bank-local mixed-precision cost; conversion and scale "
                     "handling are additive to INT8 MAC, activation broadcast uses measured "
                     "Host-to-bank bandwidth; no RTL, refresh or bank-conflict trace"),
    }


def phase_latency_projection(*, measured_seconds: float, prefill_seconds: float,
                             host_forward_ms_per_token: float, tokens: int,
                             target_ms_per_token: float,
                             host_grammar_ms_per_token: float = 0.0) -> dict:
    """Project one measured structured call onto the target token cost."""
    values = (measured_seconds, prefill_seconds, host_forward_ms_per_token,
              host_grammar_ms_per_token, target_ms_per_token)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("finite nonnegative phase values required")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("nonnegative integer token count required")
    target_seconds = prefill_seconds + tokens * target_ms_per_token / 1000
    host_control_seconds = prefill_seconds + tokens * host_forward_ms_per_token / 1000
    host_measured_decode_seconds = max(0.0, measured_seconds - prefill_seconds)
    host_control_seconds = min(measured_seconds, host_control_seconds)
    target_seconds = min(measured_seconds, target_seconds)
    return {
        "measured_seconds": measured_seconds,
        "measured_decode_seconds": host_measured_decode_seconds,
        "host_control_seconds": host_control_seconds,
        "target_seconds": target_seconds,
        "target_over_host_control": host_control_seconds / target_seconds
        if target_seconds > 0 else None,
        "host_grammar_ms_per_token": host_grammar_ms_per_token,
        "target_ms_per_token": target_ms_per_token,
    }


def grammar_residency(weight_bytes: int, token_table_bytes: int,
                      grammar_state_bytes: int, scratch_bytes: int,
                      config: StructuredControlConfig | None = None) -> dict:
    """Charge the model, token table, grammar state and scratch against one pool."""
    config = config or StructuredControlConfig()
    values = (weight_bytes, token_table_bytes, grammar_state_bytes, scratch_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise ValueError("nonnegative integer residency bytes required")
    total = sum(values)
    return {
        "weight_bytes": weight_bytes,
        "token_table_bytes": token_table_bytes,
        "grammar_state_bytes": grammar_state_bytes,
        "scratch_bytes": scratch_bytes,
        "total_bytes": total,
        "capacity_bytes": config.capacity_bytes,
        "headroom_bytes": config.capacity_bytes - total,
        "feasible": total <= config.capacity_bytes,
        "weight_copies": 1,
    }


def plane_major_layout_plan(rows, cols, bits=12, group_size=128,
                            row_bytes=2048, tile_rows=16,
                            bank_count=256) -> dict:
    """Map global plane-major codes onto bank-local DRAM rows.

    Each bank owns a contiguous shard of logical rows.  A shard is split into
    ``tile_rows`` tiles, and every bit plane of a tile occupies independently
    addressable DRAM rows.  The row-major comparison uses the same ``bits``
    width and the same physical row size; it is a layout comparison, not a
    BF16/FP32 weight inventory.
    """
    integer_fields = (rows, cols, bits, group_size, row_bytes, tile_rows,
                      bank_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in integer_fields):
        raise ValueError("positive integer plane-major layout fields required")
    if bits not in (8, 10, 12):
        raise ValueError("plane-major timing layout supports 8, 10 or 12 bits")
    if cols % group_size:
        raise ValueError("columns must align to group_size")

    rows_per_bank = math.ceil(rows / bank_count)
    plane_bytes_per_row = math.ceil(cols / 8)
    row_major_row_bytes = math.ceil(cols * bits / 8)
    row_major_rows_per_record = math.ceil(row_major_row_bytes / row_bytes)
    banks = []
    global_tile_id = 0
    plane_major_total = 0
    row_major_total = 0
    for bank in range(bank_count):
        start_row = bank * rows_per_bank
        bank_rows = max(0, min(rows_per_bank, rows - start_row))
        tile_count = math.ceil(bank_rows / tile_rows) if bank_rows else 0
        tiles = []
        dram_rows_by_plane = [0] * bits
        for tile_index in range(tile_count):
            tile_start = start_row + tile_index * tile_rows
            tile_row_count = min(tile_rows, bank_rows - tile_index * tile_rows)
            plane_bytes = tile_row_count * plane_bytes_per_row
            dram_rows_per_plane = math.ceil(plane_bytes / row_bytes)
            tiles.append({
                'global_tile_id': global_tile_id,
                'tile_index': tile_index,
                'start_row': tile_start,
                'end_row': tile_start + tile_row_count,
                'row_count': tile_row_count,
                'plane_bytes': plane_bytes,
                'dram_rows_per_plane': dram_rows_per_plane,
            })
            global_tile_id += 1
        if tile_count:
            dram_rows_per_plane = sum(
                tile['dram_rows_per_plane'] for tile in tiles)
            dram_rows_by_plane = [dram_rows_per_plane] * bits
        dram_rows_all_planes = sum(dram_rows_by_plane)
        bank_plane_bytes = sum(tile['plane_bytes'] for tile in tiles) * bits
        bank_row_major_rows = bank_rows * row_major_rows_per_record
        plane_major_total += dram_rows_all_planes
        row_major_total += bank_row_major_rows
        banks.append({
            'bank': bank,
            'start_row': start_row,
            'row_count': bank_rows,
            'tile_count': tile_count,
            'tiles': tiles,
            'dram_rows_by_plane': dram_rows_by_plane,
            'dram_rows_all_planes': dram_rows_all_planes,
            'plane_bytes': bank_plane_bytes,
            'row_major_full_read_rows': bank_row_major_rows,
        })

    return {
        'schema': 'plane-major-layout-plan-v1',
        'layout': 'global plane-major, bank-sharded, tile-padded',
        'rows': rows,
        'cols': cols,
        'bits': bits,
        'group_size': group_size,
        'row_bytes': row_bytes,
        'tile_rows': tile_rows,
        'bank_count': bank_count,
        'rows_per_bank': rows_per_bank,
        'plane_bytes_per_row': plane_bytes_per_row,
        'row_major_logical_row_bytes': row_major_row_bytes,
        'row_major_rows_per_record': row_major_rows_per_record,
        'tile_count': global_tile_id,
        'banks': banks,
        'plane_major_full_read_row_activations': plane_major_total,
        'row_major_full_read_row_activations': row_major_total,
        'plane_major_over_row_major_activations': (
            plane_major_total / row_major_total if row_major_total else None),
        'full_precision_definition': (
            'all bits planes at the same width in plane-major versus row-major; '
            'not a BF16 or FP32 source-weight inventory'),
        'boundary': (
            'analytical DRAM row layout; no RTL, refresh, repair, or '
            'runtime migration trace'),
    }
