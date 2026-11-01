"""Bank-row-owned GEMV reference and conservative, uncalibrated phase costs."""
from __future__ import annotations

import math
from functools import reduce
from math import gcd
import numpy as np

from .lpddr_bank_pim import BankPimConfig, BankPimModel


def round_bf16(value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def bank_tiled_gemv(weights, activation, *, banks=32, input_tile=256, bias=None,
                    operand_dtype='fp32'):
    if operand_dtype not in {'fp32', 'bf16'}:
        raise ValueError('operand_dtype must be fp32 or bf16')
    weights = np.asarray(weights, dtype=np.float32)
    activation = np.asarray(activation, dtype=np.float32)
    if weights.ndim != 2 or activation.shape != (weights.shape[1],):
        raise ValueError("GEMV requires [outputs, inputs] weights and [inputs] activation")
    if min(*weights.shape, banks, input_tile) <= 0:
        raise ValueError("positive dimensions and tile sizes required")
    if not np.isfinite(weights).all() or not np.isfinite(activation).all():
        raise ValueError("finite functional inputs required")
    if operand_dtype == 'bf16':
        weights = round_bf16(weights)
        activation = round_bf16(activation)
    output = np.zeros(weights.shape[0], dtype=np.float32)
    for bank in range(min(banks, weights.shape[0])):
        bank_output = output[bank::banks]
        for start in range(0, weights.shape[1], input_tile):
            end = min(start + input_tile, weights.shape[1])
            bank_output += weights[bank::banks, start:end] @ activation[start:end]
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != output.shape or not np.isfinite(bias).all():
            raise ValueError("finite bias of output dimension required")
        if operand_dtype == 'bf16':
            bias = round_bf16(bias)
        output += bias
    return output


def gemv_cost(inputs, outputs, element_bytes, config: BankPimConfig, *, input_tile=256):
    if min(inputs, outputs, element_bytes, input_tile) <= 0:
        raise ValueError("positive GEMV dimensions required")
    model = BankPimModel(config)
    counts = tuple(outputs // config.bank_count + int(bank < outputs % config.bank_count)
                   for bank in range(config.bank_count))
    active_banks = min(outputs, config.bank_count)
    tile_count = math.ceil(inputs / input_tile)
    padded_inputs = tile_count * input_tile
    resident_bytes = outputs * padded_inputs * element_bytes
    buffer_bytes = max(counts) * 4 + input_tile * element_bytes * 2
    weight_bytes = inputs * outputs * element_bytes
    host_external = weight_bytes + inputs * element_bytes + outputs * 4
    host_memory = max(model.transfer_cycles(host_external),
                      model._bank_memory(counts, inputs * element_bytes)[0])
    host_compute = math.ceil(inputs * outputs / config.host_macs_per_cycle)
    host_active = config.host_command_cycles + max(host_memory, host_compute)
    bank_memory = 0
    for start in range(0, inputs, input_tile):
        width = min(input_tile, inputs - start)
        bank_memory += model._bank_memory(counts, width * element_bytes)[0]
    bank_compute = math.ceil(max(counts) * inputs / config.pim_macs_per_bank_cycle)
    broadcast_bytes = inputs * element_bytes * active_banks
    result_bytes = outputs * 4
    command_cycles = tile_count * config.host_command_cycles + config.host_fence_cycles
    transfer_cycles = model.transfer_cycles(broadcast_bytes + result_bytes)
    pim_active = bank_memory + bank_compute + command_cycles + transfer_cycles
    host_cycles = model.refresh_elapsed(host_active)
    pim_cycles = model.refresh_elapsed(pim_active)
    stage_cycles = model.copy_cost(resident_bytes)["cycles"]
    gain = host_cycles - pim_cycles
    return {
        "inputs": inputs, "outputs": outputs, "element_bytes": element_bytes,
        "resident_bytes": resident_bytes, "local_buffer_bytes_per_bank": buffer_bytes,
        "capacity_feasible": resident_bytes <= config.pim_capacity_bytes,
        "buffer_feasible": buffer_bytes <= config.local_buffer_bytes_per_bank,
        "host_cycles": host_cycles, "pim_cycles": pim_cycles,
        "stage_cycles": stage_cycles,
        "break_even_calls": math.ceil(stage_cycles / gain) if gain > 0 else None,
        "host_external_bytes": host_external,
        "pim_external_bytes": broadcast_bytes + result_bytes,
        "pim_weight_internal_bytes": resident_bytes,
        "pim_bank_memory_cycles": bank_memory, "pim_bank_compute_cycles": bank_compute,
        "pim_command_fence_cycles": command_cycles,
        "pim_transfer_cycles": transfer_cycles,
        "pim_refresh_cycles": pim_cycles - pim_active,
        "layout": "output row r owned by bank r % banks; input tiles streamed; FP32 local accumulators",
        "modeled_operand_bytes": element_bytes, "modeled_accumulator_bytes": 4,
    }


def select_exact_capacity(candidates, capacity_bytes):
    if not isinstance(capacity_bytes, int) or capacity_bytes < 0:
        raise ValueError('exact placement requires an integer nonnegative byte budget')
    if any(not isinstance(cost[field], int) or cost[field] < minimum
           for cost in candidates.values()
           for field, minimum in (('resident_bytes', 1), ('host_cycles', 0), ('pim_cycles', 0))):
        raise ValueError('positive integer sizes and nonnegative integer cycles required')
    eligible = sorted(name for name, cost in candidates.items()
                if cost['capacity_feasible'] and cost['buffer_feasible']
                and cost['resident_bytes'] <= capacity_bytes
                and cost['host_cycles'] > cost['pim_cycles'])
    if not eligible:
        return set(), 0, 0
    unit = reduce(gcd, (candidates[name]['resident_bytes'] for name in eligible))
    capacity_units = capacity_bytes // unit
    if capacity_units > 1_000_000 or capacity_units * len(eligible) > 20_000_000:
        raise ValueError('exact integer DP state space exceeds declared resource bound; no heuristic fallback')
    best_gain = [0] * (capacity_units + 1)
    best_mask = [0] * (capacity_units + 1)
    for index, name in enumerate(eligible):
        cost = candidates[name]
        size = cost['resident_bytes'] // unit
        gain = cost['host_cycles'] - cost['pim_cycles']
        for used in range(capacity_units, size - 1, -1):
            predecessor = used - size
            value = best_gain[predecessor] + gain
            if value > best_gain[used]:
                best_gain[used] = value
                best_mask[used] = best_mask[predecessor] | (1 << index)
    selected = {name for index, name in enumerate(eligible)
                if best_mask[capacity_units] & (1 << index)}
    used = sum(candidates[name]['resident_bytes'] for name in selected)
    if (used > capacity_bytes or sum(candidates[name]['host_cycles'] - candidates[name]['pim_cycles']
                                     for name in selected) != best_gain[capacity_units]):
        raise ValueError('exact placement reconstruction failed')
    return selected, used, best_gain[capacity_units]


def project_trace(operators, config: BankPimConfig, *, input_tile=256):
    candidates = {}
    for operator in operators:
        if operator["phase"] != "decode" or operator["rows"] != 1:
            continue
        expected = operator["in_features"] * operator["out_features"] * operator["element_bytes"]
        if expected != operator["weight_bytes"] or operator["calls"] <= 0:
            raise ValueError("invalid executed linear trace")
        cost = gemv_cost(operator["in_features"], operator["out_features"],
                         operator["element_bytes"], config, input_tile=input_tile)
        if operator["name"] in candidates and candidates[operator["name"]] != cost:
            raise ValueError("operator shape changes require separate physical identities")
        candidates[operator["name"]] = cost
    selected, used, placement_gain = select_exact_capacity(candidates, config.pim_capacity_bytes)
    host_cycles = hybrid_cycles = host_bytes = hybrid_bytes = calls = 0
    per_operator = []
    for operator in operators:
        if operator["phase"] != "decode" or operator["rows"] != 1:
            continue
        name, count = operator["name"], operator["calls"]
        cost = candidates[name]
        route = "pim" if name in selected else "host"
        host_cycles += count * cost["host_cycles"]
        hybrid_cycles += count * cost[f"{route}_cycles"]
        host_bytes += count * cost["host_external_bytes"]
        hybrid_bytes += count * cost[f"{route}_external_bytes"]
        calls += count
        per_operator.append({"name": name, "calls": count, "route": route, **cost})
    stage = sum(candidates[name]["stage_cycles"] for name in selected)
    return {
        "executed_decode_linear_calls": calls, "candidate_matrices": len(candidates),
        "resident_matrices": len(selected), "resident_bytes": used,
        "capacity_bytes": config.pim_capacity_bytes,
        "host_linear_cycles": host_cycles, "warm_hybrid_linear_cycles": hybrid_cycles,
        "one_time_placement_cycles": stage, "cold_hybrid_linear_cycles": hybrid_cycles + stage,
        "warm_linear_speedup_scenario": host_cycles / hybrid_cycles if hybrid_cycles else None,
        "cold_linear_speedup_scenario": host_cycles / (hybrid_cycles + stage) if hybrid_cycles else None,
        "host_external_bytes": host_bytes, "warm_hybrid_external_bytes": hybrid_bytes,
        "end_to_end_speedup": None, "per_operator": per_operator,
        "placement_policy": "exact 0/1 capacity optimization maximizing per-call modeled cycle gain; no future-hit, output-length or call-count oracle",
        "placement_objective_gain_per_call": placement_gain,
        "placement_optimality": "exact for the declared per-call gain and shared-capacity objective",
        "schedule_contract": "serialize dependent linear calls on one shared memory resource; no free host/PIM overlap",
        "unmodeled": ["attention", "recurrent state", "vision/prefill", "nonlinear kernels",
                      "other SoC traffic", "total device DRAM capacity", "energy and thermals"],
        "boundary": "uncalibrated linear-only phase scenario; not measured PIM, cycle-accurate simulation, or E2E speedup",
    }
