"""Cycle-level bank timeline for the HyperPIM-SC decoder.

The model separates bank-local memory service, integer MAC service, row
activation, grouped-scale handling, residual refinement, I/O and refresh.  It
is deliberately a target-architecture model: GeniePIM matrix costs are used as
an external calibration point, not as a substitute for the timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
