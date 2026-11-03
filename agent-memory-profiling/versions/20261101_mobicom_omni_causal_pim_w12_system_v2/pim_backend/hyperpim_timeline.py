"""Cycle-level bank timeline for the HyperPIM-SC decoder.

The model separates bank-local memory service, integer MAC service, row
activation, grouped-scale handling, residual refinement, I/O and refresh.  It
is deliberately a target-architecture model: GeniePIM matrix costs are used as
an external calibration point, not as a substitute for the timeline.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import math


@dataclass(frozen=True)
class HyperPimTimelineConfig:
    stacks: int = 2
    channels_per_stack: int = 8
    banks_per_channel: int = 16
    clock_hz: float = 800_000_000.0
    bank_internal_bytes_per_cycle: int = 16
    int8_macs_per_bank_cycle: int = 4
    bf16_macs_per_bank_cycle: int = 2
    row_bytes: int = 2048
    t_rcd_cycles: int = 15
    t_rp_cycles: int = 15
    t_rrd_cycles: int = 4
    t_faw_cycles: int = 16
    refresh_interval_cycles: int = 3125
    refresh_block_cycles: int = 224
    matrix_command_cycles: int = 64
    layer_fence_cycles: int = 400
    segment_fences_per_layer: int = 4
    conversion_lanes_per_bank: int = 16
    bf16_reconstruction_lanes_per_bank: int = 16
    bf16_round_lanes_per_bank: int = 16
    bf16_group_reduction_lanes_per_bank: int = 16
    group_size: int = 128
    scale_bytes: int = 4
    weight_bits: int = 8
    residual_bits: int = 0
    residual_scale_bytes: int = 4
    residual_mac_relative_cost: float = 1.0
    bf16_reconstruct_weights: bool = False
    pipeline_bf16_reconstruction_and_mac: bool = False
    activation_bytes: int = 2
    output_bytes: int = 4
    external_bandwidth_bytes_per_second: float = 90_069_343_391.2629
    serialize_bank_memory_and_mac: bool = True

    def __post_init__(self) -> None:
        positive_integers = (
            self.stacks, self.channels_per_stack, self.banks_per_channel,
            self.bank_internal_bytes_per_cycle, self.int8_macs_per_bank_cycle,
            self.bf16_macs_per_bank_cycle,
            self.row_bytes, self.refresh_interval_cycles,
            self.refresh_block_cycles, self.matrix_command_cycles,
            self.layer_fence_cycles, self.segment_fences_per_layer,
            self.conversion_lanes_per_bank, self.group_size, self.scale_bytes,
            self.bf16_reconstruction_lanes_per_bank,
            self.bf16_round_lanes_per_bank,
            self.bf16_group_reduction_lanes_per_bank,
            self.weight_bits, self.residual_scale_bytes, self.activation_bytes,
            self.output_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in positive_integers):
            raise ValueError("positive integer timeline fields required")
        nonnegative_integers = (
            self.t_rcd_cycles, self.t_rp_cycles, self.t_rrd_cycles,
            self.t_faw_cycles,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in nonnegative_integers):
            raise ValueError("nonnegative integer timing fields required")
        if self.weight_bits not in (8, 10, 12):
            raise ValueError("weight bits must be eight, ten or twelve")
        if self.residual_bits not in (0, 2, 4):
            raise ValueError("residual bits must be zero, two or four")
        if self.residual_bits and self.weight_bits != 8:
            raise ValueError("residual refinement is defined only for the 8-bit base plane")
        if not isinstance(self.bf16_reconstruct_weights, bool):
            raise ValueError("BF16 reconstruction switch must be boolean")
        if not isinstance(self.pipeline_bf16_reconstruction_and_mac, bool):
            raise ValueError("BF16 pipeline switch must be boolean")
        if self.bf16_reconstruct_weights and self.weight_bits == 8:
            raise ValueError("BF16 reconstruction requires a wider-than-INT8 weight plane")
        if (self.pipeline_bf16_reconstruction_and_mac
                and not self.bf16_reconstruct_weights):
            raise ValueError("BF16 pipeline requires BF16 weight reconstruction")
        if self.refresh_block_cycles >= self.refresh_interval_cycles:
            raise ValueError("refresh blocking must leave active cycles")
        if (not math.isfinite(self.clock_hz) or self.clock_hz <= 0
                or not math.isfinite(self.external_bandwidth_bytes_per_second)
                or self.external_bandwidth_bytes_per_second <= 0
                or not math.isfinite(self.residual_mac_relative_cost)
                or self.residual_mac_relative_cost < 1):
            raise ValueError("positive finite clock/bandwidth and residual MAC cost >= 1 required")

    @property
    def bank_count(self) -> int:
        return self.stacks * self.channels_per_stack * self.banks_per_channel

    @property
    def external_bytes_per_cycle(self) -> float:
        return self.external_bandwidth_bytes_per_second / self.clock_hz


def refresh_elapsed(active_cycles: int, start_cycle: int,
                    config: HyperPimTimelineConfig) -> dict:
    """Return active + blocked cycles for one bank-synchronized refresh stream."""
    if isinstance(active_cycles, bool) or not isinstance(active_cycles, int) or active_cycles < 0:
        raise ValueError("nonnegative integer active cycles required")
    if isinstance(start_cycle, bool) or not isinstance(start_cycle, int) or start_cycle < 0:
        raise ValueError("nonnegative integer start cycle required")
    if active_cycles == 0:
        return {"active_cycles": 0, "refresh_cycles": 0, "elapsed_cycles": 0}
    interval = config.refresh_interval_cycles
    blocked = config.refresh_block_cycles
    available = interval - blocked
    offset = start_cycle % interval
    if offset >= available:
        initial_wait = interval - offset
        first_available = available
    else:
        initial_wait = 0
        first_available = available - offset
    if active_cycles <= first_available:
        elapsed = initial_wait + active_cycles
    else:
        remainder = active_cycles - first_available
        full_windows, tail = divmod(remainder - 1, available)
        elapsed = (initial_wait + first_available + blocked
                   + full_windows * interval + tail + 1)
    return {
        "active_cycles": active_cycles,
        "refresh_cycles": elapsed - active_cycles,
        "elapsed_cycles": elapsed,
    }


def matrix_cycles(inputs: int, outputs: int,
                  config: HyperPimTimelineConfig | None = None) -> dict:
    """Model one batch-1 output-row-owned matrix command window."""
    config = config or HyperPimTimelineConfig()
    if (isinstance(inputs, bool) or not isinstance(inputs, int) or inputs <= 0
            or isinstance(outputs, bool) or not isinstance(outputs, int) or outputs <= 0):
        raise ValueError("positive integer matrix dimensions required")
    active_banks = min(outputs, config.bank_count)
    rows_per_bank = math.ceil(outputs / active_banks)
    elements_per_bank = rows_per_bank * inputs
    scale_groups_per_bank = rows_per_bank * math.ceil(inputs / config.group_size)

    weight_bytes_per_bank = math.ceil(
        elements_per_bank * config.weight_bits / 8)
    memory_cycles = math.ceil(
        weight_bytes_per_bank / config.bank_internal_bytes_per_cycle)
    int8_mac_cycles = math.ceil(
        elements_per_bank / config.int8_macs_per_bank_cycle)
    mac_cycles = math.ceil(
        elements_per_bank * config.weight_bits
        / (8 * config.int8_macs_per_bank_cycle))
    if config.bf16_reconstruct_weights:
        bf16_mac_cycles = math.ceil(
            elements_per_bank / config.bf16_macs_per_bank_cycle)
        reconstruction_cycles = math.ceil(
            elements_per_bank
            / config.bf16_reconstruction_lanes_per_bank)
        rounding_cycles = math.ceil(
            elements_per_bank / config.bf16_round_lanes_per_bank)
        group_reduction_cycles = math.ceil(
            elements_per_bank
            / config.bf16_group_reduction_lanes_per_bank)
        effective_mac_cycles = bf16_mac_cycles
    else:
        bf16_mac_cycles = 0
        reconstruction_cycles = 0
        rounding_cycles = 0
        group_reduction_cycles = 0
        effective_mac_cycles = mac_cycles
    row_activations = math.ceil(elements_per_bank / config.row_bytes)
    row_cycles = row_activations * (config.t_rcd_cycles + config.t_rp_cycles)
    bank_groups = (config.stacks * config.channels_per_stack
                   * (config.banks_per_channel // 4))
    activations_per_group = math.ceil(
        row_activations * active_banks / max(1, bank_groups))
    faw_cycles = math.ceil(activations_per_group / 4) * config.t_faw_cycles
    rrd_cycles = activations_per_group * config.t_rrd_cycles
    row_cycles = max(row_cycles, faw_cycles, rrd_cycles)
    conversion_cycles = math.ceil(
        weight_bytes_per_bank / config.conversion_lanes_per_bank)
    scale_read_cycles = math.ceil(
        scale_groups_per_bank * config.scale_bytes
        / config.bank_internal_bytes_per_cycle)
    scale_apply_cycles = math.ceil(
        scale_groups_per_bank / config.conversion_lanes_per_bank)

    preparation_cycles = (
        memory_cycles + conversion_cycles + scale_read_cycles
        + scale_apply_cycles + reconstruction_cycles + rounding_cycles
        + group_reduction_cycles)
    if config.pipeline_bf16_reconstruction_and_mac:
        compute_cycles = max(preparation_cycles, effective_mac_cycles)
        overlapped_cycles = preparation_cycles + effective_mac_cycles - compute_cycles
        preparation_cycles_charged = 0
    else:
        if config.serialize_bank_memory_and_mac:
            compute_cycles = memory_cycles + effective_mac_cycles
        else:
            compute_cycles = max(memory_cycles, effective_mac_cycles)
        overlapped_cycles = 0
        preparation_cycles_charged = (
            conversion_cycles + scale_read_cycles + scale_apply_cycles
            + reconstruction_cycles + rounding_cycles + group_reduction_cycles)

    if config.residual_bits:
        residual_payload_bytes = math.ceil(
            elements_per_bank * config.residual_bits / 8)
        residual_conversion_cycles = math.ceil(
            residual_payload_bytes / config.conversion_lanes_per_bank)
        residual_scale_read_cycles = math.ceil(
            scale_groups_per_bank * config.residual_scale_bytes
            / config.bank_internal_bytes_per_cycle)
        residual_scale_apply_cycles = math.ceil(
            scale_groups_per_bank / config.conversion_lanes_per_bank)
        residual_mac_cycles = math.ceil(
            elements_per_bank * config.residual_mac_relative_cost
            / config.int8_macs_per_bank_cycle)
    else:
        residual_payload_bytes = 0
        residual_conversion_cycles = 0
        residual_scale_read_cycles = 0
        residual_scale_apply_cycles = 0
        residual_mac_cycles = 0

    transfer_bytes = inputs * config.activation_bytes + outputs * config.output_bytes
    transfer_cycles = math.ceil(transfer_bytes / config.external_bytes_per_cycle)
    command_cycles = config.matrix_command_cycles
    active_cycles = (
        compute_cycles + row_cycles + preparation_cycles_charged
        + residual_conversion_cycles
        + residual_scale_read_cycles + residual_scale_apply_cycles
        + residual_mac_cycles + transfer_cycles + command_cycles)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "active_banks": active_banks,
        "rows_per_bank": rows_per_bank,
        "elements_per_bank": elements_per_bank,
        "weight_bits": config.weight_bits,
        "weight_bytes_per_bank": weight_bytes_per_bank,
        "scale_groups_per_bank": scale_groups_per_bank,
        "memory_cycles": memory_cycles,
        "int8_mac_cycles": int8_mac_cycles,
        "mac_cycles": mac_cycles,
        "bf16_mac_cycles": bf16_mac_cycles,
        "effective_mac_cycles": effective_mac_cycles,
        "bf16_reconstruction_cycles": reconstruction_cycles,
        "bf16_rounding_cycles": rounding_cycles,
        "bf16_group_reduction_cycles": group_reduction_cycles,
        "bf16_preparation_cycles": preparation_cycles,
        "bf16_pipeline_cycles": compute_cycles,
        "bf16_pipeline_overlap_cycles": overlapped_cycles,
        "row_activations_per_bank": row_activations,
        "row_activation_cycles": row_cycles,
        "conversion_cycles": conversion_cycles,
        "scale_read_cycles": scale_read_cycles,
        "scale_apply_cycles": scale_apply_cycles,
        "residual_payload_bytes_per_bank": residual_payload_bytes,
        "residual_conversion_cycles": residual_conversion_cycles,
        "residual_scale_read_cycles": residual_scale_read_cycles,
        "residual_scale_apply_cycles": residual_scale_apply_cycles,
        "residual_mac_cycles": residual_mac_cycles,
        "external_transfer_bytes": transfer_bytes,
        "external_transfer_cycles": transfer_cycles,
        "matrix_command_cycles": command_cycles,
        "active_cycles": active_cycles,
        "active_ms": 1000 * active_cycles / config.clock_hz,
    }


def simulate_token(sequence, config: HyperPimTimelineConfig | None = None) -> dict:
    """Simulate an ordered per-token decode Linear sequence with refresh stalls.

    Each sequence item is ``{name, inputs, outputs, count?}``.  A count models a
    repeated call with the same resident weights; the default is one.
    """
    config = config or HyperPimTimelineConfig()
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise ValueError("nonempty ordered matrix sequence required")
    current_cycle = 0
    events = []
    layer_start = {}
    layer_cycles = {}
    totals = {
        "active_cycles": 0, "refresh_cycles": 0, "elapsed_cycles": 0,
        "calls": 0, "weight_elements": 0, "weight_bytes": 0, "macs": 0,
        "row_activations": 0, "external_transfer_bytes": 0,
        "conversion_cycles": 0, "residual_conversion_cycles": 0,
        "residual_mac_cycles": 0, "matrix_command_cycles": 0,
        "bf16_reconstruction_cycles": 0, "bf16_rounding_cycles": 0,
        "bf16_group_reduction_cycles": 0, "bf16_mac_cycles": 0,
        "bf16_pipeline_overlap_cycles": 0,
    }
    for item in sequence:
        if not isinstance(item, dict):
            raise ValueError("matrix sequence items must be objects")
        name = str(item.get("name") or "")
        inputs = int(item.get("inputs") or item.get("in_features") or 0)
        outputs = int(item.get("outputs") or item.get("out_features") or 0)
        count = int(item.get("count", 1))
        if not name or count <= 0:
            raise ValueError("named positive matrix call count required")
        estimate = matrix_cycles(inputs, outputs, config)
        start = current_cycle
        for repeat in range(count):
            if repeat:
                start = current_cycle
            refreshed = refresh_elapsed(estimate["active_cycles"], start, config)
            current_cycle += refreshed["elapsed_cycles"]
            events.append({
                "name": name,
                "repeat": repeat,
                "start_cycle": start,
                "end_cycle": current_cycle,
                **estimate,
                "refresh_cycles": refreshed["refresh_cycles"],
                "elapsed_cycles": refreshed["elapsed_cycles"],
            })
            totals["refresh_cycles"] += refreshed["refresh_cycles"]
        totals["active_cycles"] += estimate["active_cycles"] * count
        totals["calls"] += count
        totals["weight_elements"] += inputs * outputs * count
        totals["weight_bytes"] += (estimate["weight_bytes_per_bank"]
                                   * estimate["active_banks"] * count)
        totals["macs"] += inputs * outputs * count * (1 + int(bool(config.residual_bits)))
        totals["row_activations"] += estimate["row_activations_per_bank"] * count
        totals["external_transfer_bytes"] += estimate["external_transfer_bytes"] * count
        totals["conversion_cycles"] += estimate["conversion_cycles"] * count
        totals["residual_conversion_cycles"] += estimate["residual_conversion_cycles"] * count
        totals["residual_mac_cycles"] += estimate["residual_mac_cycles"] * count
        totals["bf16_reconstruction_cycles"] += (
            estimate["bf16_reconstruction_cycles"] * count)
        totals["bf16_rounding_cycles"] += estimate["bf16_rounding_cycles"] * count
        totals["bf16_group_reduction_cycles"] += (
            estimate["bf16_group_reduction_cycles"] * count)
        totals["bf16_mac_cycles"] += estimate["bf16_mac_cycles"] * count
        totals["bf16_pipeline_overlap_cycles"] += (
            estimate["bf16_pipeline_overlap_cycles"] * count)
        totals["matrix_command_cycles"] += estimate["matrix_command_cycles"] * count
        layer = item.get("layer")
        if isinstance(layer, int) and layer >= 0:
            layer_start.setdefault(layer, start)
            layer_cycles[layer] = current_cycle - layer_start[layer]
    fence_cycles = 0
    if layer_cycles:
        fence_cycles = (len(layer_cycles) * config.segment_fences_per_layer
                        * config.layer_fence_cycles)
        current_cycle += fence_cycles
    totals["layer_fence_cycles"] = fence_cycles
    totals["elapsed_cycles"] = current_cycle
    totals["elapsed_ms"] = 1000 * current_cycle / config.clock_hz
    totals["refreshed_ms"] = totals["elapsed_ms"]
    totals["pim_linear_ms_per_token"] = totals["elapsed_ms"]
    return {
        "config": {
            "banks": config.bank_count,
            "clock_hz": config.clock_hz,
            "bank_internal_bytes_per_cycle": config.bank_internal_bytes_per_cycle,
            "int8_macs_per_bank_cycle": config.int8_macs_per_bank_cycle,
            "bf16_macs_per_bank_cycle": config.bf16_macs_per_bank_cycle,
            "row_bytes": config.row_bytes,
            "refresh_interval_cycles": config.refresh_interval_cycles,
            "refresh_block_cycles": config.refresh_block_cycles,
            "weight_bits": config.weight_bits,
            "residual_bits": config.residual_bits,
            "residual_mac_relative_cost": config.residual_mac_relative_cost,
            "bf16_reconstruct_weights": config.bf16_reconstruct_weights,
            "pipeline_bf16_reconstruction_and_mac": (
                config.pipeline_bf16_reconstruction_and_mac),
            "bf16_reconstruction_lanes_per_bank": (
                config.bf16_reconstruction_lanes_per_bank),
            "bf16_round_lanes_per_bank": config.bf16_round_lanes_per_bank,
            "bf16_group_reduction_lanes_per_bank": (
                config.bf16_group_reduction_lanes_per_bank),
            "serialize_bank_memory_and_mac": config.serialize_bank_memory_and_mac,
        },
        "totals": totals,
        "layers": {str(layer): {"cycles": cycles,
                                  "ms": 1000 * cycles / config.clock_hz}
                   for layer, cycles in sorted(layer_cycles.items())},
        "events": events,
        "boundary": ("target LPDDR bank-PIM cycle model calibrated against GeniePIM; "
                     "not RTL, silicon, phone measurement, or end-to-end speedup"),
    }


def with_weight_bits(config: HyperPimTimelineConfig,
                      bits: int) -> HyperPimTimelineConfig:
    if bits not in (8, 10, 12):
        raise ValueError("weight bits must be eight, ten or twelve")
    return replace(config, weight_bits=bits, residual_bits=0,
                   bf16_reconstruct_weights=bits in (10, 12),
                   pipeline_bf16_reconstruction_and_mac=bits in (10, 12))


def with_residual_bits(config: HyperPimTimelineConfig, bits: int) -> HyperPimTimelineConfig:
    if bits not in (0, 2, 4):
        raise ValueError("residual bits must be zero, two or four")
    return replace(config, residual_bits=bits)


def _timing_config(value) -> HyperPimTimelineConfig:
    if isinstance(value, HyperPimTimelineConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("timing_profile must be a timeline config or mapping")
    allowed = {field.name for field in fields(HyperPimTimelineConfig)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown timing_profile fields: {unknown}")
    return HyperPimTimelineConfig(**dict(value))


def _layout_tile_map(layout_plan):
    if not isinstance(layout_plan, Mapping):
        raise ValueError("layout_plan must be a mapping")
    tiling = {}
    banks = layout_plan.get("banks")
    if not isinstance(banks, list) or not banks:
        raise ValueError("layout_plan banks must be a nonempty list")
    for bank in banks:
        bank_id = int(bank["bank"])
        for tile in bank.get("tiles", []):
            tile_id = int(tile["global_tile_id"])
            if tile_id in tiling:
                raise ValueError("duplicate global tile identity in layout_plan")
            tiling[tile_id] = (bank_id, tile)
    return banks, tiling


def progressive_topk_cycles(layout_plan, rounds: list[dict], timing_profile,
                            host_round_trip_cycles: int | None) -> dict:
    """Estimate bit-plane rounds and compare them with a full GEMV.

    T2.1 currently exports survivor counts, not survivor row identities.  When
    a round lacks ``active_tile_ids`` this function therefore uses a clearly
    labelled count bound: survivors are distributed evenly across active banks
    and each bank's touched-tile count is ``ceil(rows_per_bank / tile_rows)``.
    """
    if (host_round_trip_cycles is None
            or isinstance(host_round_trip_cycles, bool)
            or not isinstance(host_round_trip_cycles, int)
            or host_round_trip_cycles < 0):
        raise ValueError("host_round_trip_cycles must be a nonnegative integer")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("nonempty progressive top-k rounds required")
    config = _timing_config(timing_profile)
    banks, tile_map = _layout_tile_map(layout_plan)
    bits = int(layout_plan.get("bits") or 0)
    cols = int(layout_plan.get("cols") or 0)
    rows = int(layout_plan.get("rows") or 0)
    tile_rows = int(layout_plan.get("tile_rows") or 0)
    bank_count = int(layout_plan.get("bank_count") or 0)
    row_bytes = int(layout_plan.get("row_bytes") or 0)
    if (bits not in (8, 10, 12) or cols <= 0 or rows <= 0 or tile_rows <= 0
            or bank_count <= 0 or row_bytes <= 0):
        raise ValueError("layout_plan is missing valid plane/geometry fields")
    if len(banks) != bank_count:
        raise ValueError("layout_plan bank count does not match its bank list")

    plane_total = 0
    result_rounds = []
    current_cycle = 0
    total_row_activations = 0
    total_read_cycles = 0
    total_mac_cycles = 0
    total_bound_cycles = 0
    total_threshold_cycles = 0
    total_internal_read_bytes = 0
    total_partial_macs = 0
    total_bound_updates = 0
    any_tile_ids = False
    for round_index, item in enumerate(rounds):
        if not isinstance(item, Mapping):
            raise ValueError("progressive top-k rounds must be mappings")
        planes_in_round = item.get("planes_in_round")
        rows_alive = item.get("rows_alive_before")
        if (isinstance(planes_in_round, bool)
                or not isinstance(planes_in_round, int)
                or planes_in_round <= 0
                or isinstance(rows_alive, bool)
                or not isinstance(rows_alive, int)
                or rows_alive <= 0):
            raise ValueError("round plane and survivor counts must be positive")
        if rows_alive > rows:
            raise ValueError("round survivor count exceeds layout rows")
        plane_total += planes_in_round
        if plane_total > bits:
            raise ValueError("progressive rounds exceed layout planes")

        active_tile_ids = item.get("active_tile_ids")
        if active_tile_ids is not None:
            any_tile_ids = True
            ids = list(active_tile_ids)
            if (not ids or any(isinstance(value, bool)
                               or not isinstance(value, int) for value in ids)
                    or len(ids) != len(set(ids))
                    or not set(ids) <= set(tile_map)):
                raise ValueError("active_tile_ids must be unique known integers")
            per_bank = {}
            for tile_id in ids:
                bank_id, _ = tile_map[tile_id]
                per_bank[bank_id] = per_bank.get(bank_id, 0) + 1
            active_banks = len(per_bank)
            tiles_per_active_bank = max(per_bank.values())
            bank_plane_rows = {}
            for tile_id in ids:
                bank_id, tile = tile_map[tile_id]
                bank_plane_rows[bank_id] = (
                    bank_plane_rows.get(bank_id, 0)
                    + tile["dram_rows_per_plane"])
            per_bank_plane_rows = max(bank_plane_rows.values())
            total_bank_activations = sum(bank_plane_rows.values())
            tile_source = "provided_active_tile_ids"
        else:
            active_banks = min(bank_count, rows_alive)
            rows_per_active_bank = math.ceil(rows_alive / active_banks)
            tiles_per_active_bank = math.ceil(rows_per_active_bank / tile_rows)
            touched_banks = banks[:active_banks]
            per_bank_plane_rows = max(
                sum(tile["dram_rows_per_plane"]
                    for tile in bank.get("tiles", [])[:tiles_per_active_bank])
                for bank in touched_banks)
            total_bank_activations = sum(
                sum(tile["dram_rows_per_plane"]
                    for tile in bank.get("tiles", [])[:tiles_per_active_bank])
                for bank in touched_banks)
            tile_source = "count_bound_no_row_ids"

        row_activations_per_bank = per_bank_plane_rows * planes_in_round
        row_activations_total = total_bank_activations * planes_in_round
        read_bytes_per_bank = row_activations_per_bank * row_bytes
        read_cycles = math.ceil(
            read_bytes_per_bank / config.bank_internal_bytes_per_cycle)
        rows_per_active_bank = math.ceil(rows_alive / active_banks)
        elements_per_bank = rows_per_active_bank * cols
        mac_cycles = math.ceil(
            elements_per_bank * planes_in_round
            / (8 * config.int8_macs_per_bank_cycle))
        bound_cycles = math.ceil(
            2 * elements_per_bank / config.int8_macs_per_bank_cycle)
        active_cycles = (read_cycles + mac_cycles + bound_cycles
                         + host_round_trip_cycles)
        refreshed = refresh_elapsed(active_cycles, current_cycle, config)
        result_rounds.append({
            "round": round_index,
            "planes_in_round": planes_in_round,
            "planes_read_total": plane_total,
            "rows_alive_before": rows_alive,
            "active_banks": active_banks,
            "tiles_per_active_bank": tiles_per_active_bank,
            "tile_identity_source": tile_source,
            "row_activations_per_bank": row_activations_per_bank,
            "row_activations_total": row_activations_total,
            "internal_read_bytes_per_bank": read_bytes_per_bank,
            "internal_read_cycles": read_cycles,
            "mac_cycles": mac_cycles,
            "bound_cycles": bound_cycles,
            "threshold_broadcast_cycles": host_round_trip_cycles,
            "active_cycles": active_cycles,
            "start_cycle": current_cycle,
            "refresh_cycles": refreshed["refresh_cycles"],
            "elapsed_cycles": refreshed["elapsed_cycles"],
        })
        current_cycle += refreshed["elapsed_cycles"]
        total_row_activations += row_activations_total
        total_read_cycles += read_cycles
        total_mac_cycles += mac_cycles
        total_bound_cycles += bound_cycles
        total_threshold_cycles += host_round_trip_cycles
        total_internal_read_bytes += read_bytes_per_bank * active_banks
        total_partial_macs += (
            elements_per_bank * planes_in_round * active_banks)
        total_bound_updates += 2 * elements_per_bank * active_banks
    if plane_total != bits:
        raise ValueError("progressive rounds must read every layout plane")

    full_config = with_weight_bits(config, bits)
    full_gemv = matrix_cycles(cols, rows, full_config)
    total_active_cycles = sum(item["active_cycles"] for item in result_rounds)
    return {
        "schema": "progressive-topk-cycles-v1",
        "rows": rows,
        "cols": cols,
        "bits": bits,
        "bank_count": bank_count,
        "tile_rows": tile_rows,
        "host_round_trip_cycles": host_round_trip_cycles,
        "tile_identity_source": (
            "provided_active_tile_ids" if any_tile_ids
            else "count_bound_no_row_ids"),
        "rounds": result_rounds,
        "totals": {
            "rounds": len(result_rounds),
            "row_activations_total": total_row_activations,
            "internal_read_cycles": total_read_cycles,
            "mac_cycles": total_mac_cycles,
            "bound_cycles": total_bound_cycles,
            "threshold_broadcast_cycles": total_threshold_cycles,
            "internal_read_bytes": total_internal_read_bytes,
            "partial_macs": total_partial_macs,
            "bound_updates": total_bound_updates,
            "active_cycles": total_active_cycles,
            "refresh_cycles": current_cycle - total_active_cycles,
            "elapsed_cycles": current_cycle,
            "active_ms": 1000 * total_active_cycles / config.clock_hz,
            "elapsed_ms": 1000 * current_cycle / config.clock_hz,
        },
        "full_gemv": full_gemv,
        "active_cycles_over_full_gemv": (
            total_active_cycles / full_gemv["active_cycles"]
            if full_gemv["active_cycles"] else None),
        "boundary": (
            "analytical bit-plane round model; count-bound tile identity is "
            "used when T2.1 stats do not carry survivor row identities"),
    }


def retrieval_gemv_workload(n_records, dim=1024) -> dict:
    """Represent dense retrieval scoring as a workload for the existing model."""
    if (isinstance(n_records, bool) or not isinstance(n_records, int)
            or n_records <= 0 or isinstance(dim, bool)
            or not isinstance(dim, int) or dim <= 0):
        raise ValueError("positive integer retrieval dimensions required")
    sequence_item = {
        "name": "dense_retrieval",
        "inputs": dim,
        "outputs": n_records,
        "count": 1,
    }
    estimate = matrix_cycles(dim, n_records)
    return {
        "schema": "retrieval-gemv-workload-v1",
        "name": sequence_item["name"],
        "inputs": dim,
        "outputs": n_records,
        "macs": n_records * dim,
        "weight_elements": n_records * dim,
        "sequence_item": sequence_item,
        "timeline": estimate,
        "cycles": estimate["active_cycles"],
        "boundary": (
            "batch-1 dense retrieval workload in the existing target cycle "
            "model; not a measured PIM kernel"),
    }


ENERGY_PARAM_KEYS = (
    "e_act_pj_per_row",
    "e_read_pj_per_byte_internal",
    "e_io_pj_per_byte",
    "e_mac_bf16_pj",
    "e_bound_update_pj",
    "e_refresh_pj_per_cycle",
)
ENERGY_COUNT_KEYS = (
    "row_activations",
    "internal_read_bytes",
    "io_bytes",
    "bf16_macs",
    "bound_updates",
    "refresh_cycles",
)


def energy_model(counts: dict, energy_params: dict) -> dict:
    """Apply a fully specified energy profile to explicit execution counts."""
    if not isinstance(energy_params, Mapping):
        raise ValueError("energy_params must be a mapping")
    unknown = sorted(set(energy_params) - set(ENERGY_PARAM_KEYS))
    if unknown:
        raise ValueError(f"unknown energy parameter names: {unknown}")
    missing = [
        key for key in ENERGY_PARAM_KEYS
        if energy_params.get(key) is None
    ]
    if missing:
        return {"status": "params_missing", "missing": missing}
    if not isinstance(counts, Mapping):
        raise ValueError("counts must be a mapping")
    counts_missing = [key for key in ENERGY_COUNT_KEYS if key not in counts]
    if counts_missing:
        raise ValueError(f"energy counts are missing: {counts_missing}")
    parameters = {}
    for key in ENERGY_PARAM_KEYS:
        value = energy_params[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError("energy parameters must be finite nonnegative numbers")
        parameters[key] = float(value)
    values = {}
    for key in ENERGY_COUNT_KEYS:
        value = counts[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError("energy counts must be finite nonnegative numbers")
        values[key] = float(value)
    components = {
        "activation_pj": values["row_activations"] * parameters["e_act_pj_per_row"],
        "internal_read_pj": values["internal_read_bytes"]
        * parameters["e_read_pj_per_byte_internal"],
        "io_pj": values["io_bytes"] * parameters["e_io_pj_per_byte"],
        "mac_bf16_pj": values["bf16_macs"] * parameters["e_mac_bf16_pj"],
        "bound_update_pj": values["bound_updates"]
        * parameters["e_bound_update_pj"],
        "refresh_pj": values["refresh_cycles"]
        * parameters["e_refresh_pj_per_cycle"],
    }
    total_pj = sum(components.values())
    return {
        "status": "ok",
        "counts": values,
        "energy_params": parameters,
        "components_pj": components,
        "total_pj": total_pj,
        "total_nj": total_pj / 1000.0,
        "boundary": (
            "closed-form energy accounting over caller-provided counts and "
            "parameters; no values are inferred or measured"),
    }
