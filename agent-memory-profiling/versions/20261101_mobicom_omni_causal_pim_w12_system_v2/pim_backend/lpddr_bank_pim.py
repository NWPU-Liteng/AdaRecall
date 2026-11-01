"""Auditable phase-level LPDDR bank-PIM costs, not a Ramulator extension."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BankPimConfig:
    channels: int = 2
    banks_per_channel: int = 16
    bank_groups: int = 4
    clock_hz: float = 800_000_000.0
    external_bytes_per_channel_cycle: int = 16
    transfer_bytes: int = 32
    burst_cycles: int = 4
    column_command_cycles: int = 4
    bank_internal_bytes_per_cycle: int = 16
    row_bytes: int = 2048
    t_rcd: int = 15
    t_cas: int = 20
    t_rp: int = 15
    t_rrd: int = 4
    t_faw: int = 16
    refresh_interval: int = 3125
    refresh_cycles: int = 224
    pim_macs_per_bank_cycle: int = 16
    host_macs_per_cycle: int = 512
    host_softmax_elements_per_cycle: int = 32
    host_command_cycles: int = 1600
    host_fence_cycles: int = 400
    pim_capacity_bytes: int = 2 * 1024**3
    local_buffer_bytes_per_bank: int = 65536
    host_background_bytes_per_cycle: float = 0.0

    def __post_init__(self) -> None:
        positive = (self.channels, self.banks_per_channel, self.bank_groups,
                    self.clock_hz, self.external_bytes_per_channel_cycle,
                    self.transfer_bytes, self.burst_cycles, self.column_command_cycles,
                    self.bank_internal_bytes_per_cycle, self.row_bytes,
                    self.refresh_interval, self.pim_macs_per_bank_cycle,
                    self.host_macs_per_cycle, self.host_softmax_elements_per_cycle,
                    self.pim_capacity_bytes, self.local_buffer_bytes_per_bank)
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("dimensions, capacity and rates must be finite and positive")
        timings = (self.t_rcd, self.t_cas, self.t_rp, self.t_rrd, self.t_faw,
                   self.refresh_cycles, self.host_command_cycles, self.host_fence_cycles)
        if min(timings) < 0 or self.refresh_cycles >= self.refresh_interval:
            raise ValueError("invalid timing or refresh budget")
        if self.banks_per_channel % self.bank_groups:
            raise ValueError("bank groups must divide banks per channel")
        if not 0 <= self.host_background_bytes_per_cycle < self.external_service_bytes_per_cycle:
            raise ValueError("background traffic must leave positive external bandwidth")

    @property
    def bank_count(self) -> int:
        return self.channels * self.banks_per_channel

    @property
    def external_bytes_per_cycle(self) -> int:
        return self.channels * self.external_bytes_per_channel_cycle

    @property
    def external_service_bytes_per_cycle(self) -> float:
        command_rate = self.channels * self.transfer_bytes / max(self.burst_cycles, self.column_command_cycles)
        return min(self.external_bytes_per_cycle, command_rate)


class BankPimModel:
    def __init__(self, config: BankPimConfig) -> None:
        self.config = config

    def transfer_cycles(self, byte_count: int) -> int:
        if byte_count < 0:
            raise ValueError("negative traffic")
        cfg = self.config
        padded = math.ceil(byte_count / (cfg.channels * cfg.transfer_bytes)) * cfg.channels * cfg.transfer_bytes
        bandwidth = cfg.external_service_bytes_per_cycle - cfg.host_background_bytes_per_cycle
        return math.ceil(padded / bandwidth)

    def refresh_elapsed(self, active_cycles: int, start_cycle: int = 0) -> int:
        if min(active_cycles, start_cycle) < 0:
            raise ValueError("negative duration")
        if active_cycles == 0:
            return 0
        interval = self.config.refresh_interval
        blocked = self.config.refresh_cycles
        available = interval - blocked
        offset = start_cycle % interval
        if offset >= available:
            initial_wait = interval - offset
            first_available = available
        else:
            initial_wait = 0
            first_available = available - offset
        if active_cycles <= first_available:
            return initial_wait + active_cycles
        remainder = active_cycles - first_available
        full_windows, tail = divmod(remainder - 1, available)
        return initial_wait + first_available + blocked + full_windows * interval + tail + 1

    def copy_cost(self, byte_count: int) -> dict:
        active = self.transfer_cycles(byte_count)
        elapsed = self.refresh_elapsed(active)
        return {"external_bytes": byte_count, "active_cycles": active,
                "refresh_cycles": elapsed - active, "cycles": elapsed}

    def _bank_memory(self, counts: tuple[int, ...], bytes_per_record: int) -> tuple[int, int]:
        cfg = self.config
        max_bytes = max(counts, default=0) * bytes_per_record
        rows = [math.ceil(count * bytes_per_record / cfg.row_bytes) for count in counts]
        max_rows = max(rows, default=0)
        service = (math.ceil(max_bytes / cfg.bank_internal_bytes_per_cycle)
                   + max_rows * (cfg.t_rcd + cfg.t_cas)
                   + max(0, max_rows - 1) * cfg.t_rp)
        channel_rows = [sum(rows[channel::cfg.channels]) for channel in range(cfg.channels)]
        activation_issue = max(channel_rows, default=0) * max(cfg.t_rrd, math.ceil(cfg.t_faw / 4))
        return max(service, activation_issue), sum(rows)

    def attention_step(self, context_tokens: int, *, query_heads: int,
                       kv_heads: int, head_dim: int, element_bytes: int = 2,
                       accumulator_bytes: int = 4, route: str = "pim") -> dict:
        if min(context_tokens, query_heads, kv_heads, head_dim, element_bytes, accumulator_bytes) <= 0:
            raise ValueError("positive attention dimensions required")
        if query_heads % kv_heads or route not in {"host", "pim"}:
            raise ValueError("invalid GQA grouping or route")
        cfg = self.config
        counts = tuple(context_tokens // cfg.bank_count + int(bank < context_tokens % cfg.bank_count)
                       for bank in range(cfg.bank_count))
        active_banks = min(context_tokens, cfg.bank_count)
        kv_half = context_tokens * kv_heads * head_dim * element_bytes
        query_bytes = query_heads * head_dim * element_bytes
        output_bytes = query_heads * head_dim * accumulator_bytes
        scores_bytes = context_tokens * query_heads * accumulator_bytes
        macs_half = context_tokens * query_heads * head_dim
        bank_memory, activations = self._bank_memory(counts, kv_heads * head_dim * element_bytes)
        softmax_cycles = math.ceil(context_tokens * query_heads / cfg.host_softmax_elements_per_cycle)
        phases = []
        cursor = 0

        def phase(name: str, active_cycles: int, external_bytes: int = 0,
                  internal_bytes: int = 0, macs: int = 0, memory_cycles: int = 0,
                  compute_cycles: int = 0, commands: int = 0, fences: int = 0) -> None:
            nonlocal cursor
            elapsed = self.refresh_elapsed(active_cycles, cursor)
            phases.append({"name": name, "start_cycle": cursor, "cycles": elapsed,
                           "active_cycles": active_cycles, "refresh_cycles": elapsed - active_cycles,
                           "external_bytes": external_bytes, "bank_internal_bytes": internal_bytes,
                           "macs": macs, "memory_cycles": memory_cycles,
                           "compute_cycles": compute_cycles, "host_commands": commands,
                           "host_fences": fences})
            cursor += elapsed

        if route == "host":
            external_bytes = 2 * kv_half + query_bytes + output_bytes
            memory_cycles = max(self.transfer_cycles(external_bytes), 2 * bank_memory)
            compute_cycles = math.ceil(2 * macs_half / cfg.host_macs_per_cycle) + softmax_cycles
            phase("host_fused_streaming_attention", cfg.host_command_cycles + max(memory_cycles, compute_cycles),
                  external_bytes, macs=2 * macs_half, memory_cycles=memory_cycles,
                  compute_cycles=compute_cycles, commands=1)
        else:
            bank_compute = math.ceil(max(counts) * query_heads * head_dim / cfg.pim_macs_per_bank_cycle)
            phase("pim_qk_submit_and_query_broadcast",
                  cfg.host_command_cycles + self.transfer_cycles(query_bytes * active_banks),
                  query_bytes * active_banks, commands=1)
            phase("bank_qk", bank_memory + bank_compute, internal_bytes=kv_half,
                  macs=macs_half, memory_cycles=bank_memory, compute_cycles=bank_compute)
            phase("qk_fence_and_score_return", cfg.host_fence_cycles + self.transfer_cycles(scores_bytes),
                  scores_bytes, fences=1)
            phase("host_softmax", cfg.host_command_cycles + softmax_cycles,
                  compute_cycles=softmax_cycles, commands=1)
            phase("pim_av_submit_and_probability_delivery",
                  cfg.host_command_cycles + self.transfer_cycles(scores_bytes), scores_bytes, commands=1)
            phase("bank_av", bank_memory + bank_compute, internal_bytes=kv_half,
                  macs=macs_half, memory_cycles=bank_memory, compute_cycles=bank_compute)
            partial_bytes = output_bytes * active_banks
            phase("av_fence_and_partial_return", cfg.host_fence_cycles + self.transfer_cycles(partial_bytes),
                  partial_bytes, fences=1)
            reduction_ops = max(0, active_banks - 1) * query_heads * head_dim
            phase("host_partial_reduce", cfg.host_command_cycles + math.ceil(reduction_ops / cfg.host_macs_per_cycle),
                  compute_cycles=math.ceil(reduction_ops / cfg.host_macs_per_cycle), commands=1)
        return {"route": route, "cycles": cursor, "logical_kv_read_bytes": 2 * kv_half,
                "external_bytes": sum(item["external_bytes"] for item in phases),
                "bank_internal_bytes": sum(item["bank_internal_bytes"] for item in phases),
                "macs": 2 * macs_half, "row_activations": 2 * activations,
                "host_commands": sum(item["host_commands"] for item in phases),
                "host_fences": sum(item["host_fences"] for item in phases),
                "bank_token_counts": counts, "phases": phases}

    def admission(self, *, resident_bytes: int, query_heads: int, head_dim: int,
                  element_bytes: int = 2) -> dict:
        if resident_bytes < 0 or min(query_heads, head_dim, element_bytes) <= 0:
            raise ValueError("negative capacity request")
        workspace = query_heads * head_dim * (element_bytes + 4) + query_heads * 4
        reasons = []
        if resident_bytes > self.config.pim_capacity_bytes:
            reasons.append("resident_capacity")
        if workspace > self.config.local_buffer_bytes_per_bank:
            reasons.append("bank_workspace")
        return {"admitted": not reasons, "reasons": reasons, "resident_bytes": resident_bytes,
                "bank_workspace_bytes": workspace}

    def resident_score_scan(self, vector_count: int, dimension: int, *, top_k: int,
                            route: str = "pim", element_bytes: int = 1) -> dict:
        if (min(vector_count, dimension, top_k, element_bytes) <= 0
                or route not in {"host", "pim"}):
            raise ValueError("invalid resident score scan")
        cfg = self.config
        counts = tuple(vector_count // cfg.bank_count + int(bank < vector_count % cfg.bank_count)
                       for bank in range(cfg.bank_count))
        active_banks = min(vector_count, cfg.bank_count)
        resident_bytes = vector_count * dimension * element_bytes
        admission = {"admitted": resident_bytes <= cfg.pim_capacity_bytes
                     and dimension * element_bytes + 8 * min(top_k, vector_count)
                     <= cfg.local_buffer_bytes_per_bank}
        if route == "pim" and not admission["admitted"]:
            result = self.resident_score_scan(vector_count, dimension, top_k=top_k,
                                              route="host", element_bytes=element_bytes)
            return {**result, "route": "host_capacity_fallback", "admission": admission}
        bytes_per_record = dimension * element_bytes
        memory_cycles, activations = self._bank_memory(counts, bytes_per_record)
        macs = vector_count * dimension
        if route == "host":
            external_bytes = resident_bytes + bytes_per_record + min(top_k, vector_count) * 8
            compute = math.ceil(macs / cfg.host_macs_per_cycle)
            topk = math.ceil(vector_count * math.ceil(math.log2(min(top_k, vector_count) + 1))
                             / cfg.host_softmax_elements_per_cycle)
            active_cycles = cfg.host_command_cycles + max(self.transfer_cycles(external_bytes), memory_cycles, compute + topk)
        else:
            candidates = sum(min(top_k, count) for count in counts)
            external_bytes = bytes_per_record * active_banks + candidates * 8
            compute = math.ceil(max(counts) * dimension / cfg.pim_macs_per_bank_cycle)
            topk = max(counts) * math.ceil(math.log2(min(top_k, vector_count) + 1))
            merge = math.ceil(candidates * math.ceil(math.log2(min(top_k, vector_count) + 1))
                              / cfg.host_softmax_elements_per_cycle)
            active_cycles = (2 * cfg.host_command_cycles + cfg.host_fence_cycles
                             + self.transfer_cycles(external_bytes) + memory_cycles + compute + topk + merge)
        cycles = self.refresh_elapsed(active_cycles)
        return {"route": route, "admission": admission, "cycles": cycles,
                "active_cycles": active_cycles, "refresh_cycles": cycles - active_cycles,
                "external_bytes": external_bytes, "logical_read_bytes": resident_bytes,
                "bank_internal_bytes": resident_bytes if route == "pim" else 0,
                "macs": macs, "row_activations": activations,
                "host_commands": 1 if route == "host" else 2,
                "boundary": f"resident scoring/top-k primitive with {element_bytes}-byte operands; not the frozen cross-encoder"}


def bank_sharded_attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray,
                           *, banks: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    keys = np.asarray(keys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if (banks <= 0 or query.ndim != 2 or keys.ndim != 3 or keys.shape != values.shape
            or min(keys.shape) <= 0 or min(query.shape) <= 0 or query.shape[1] != keys.shape[2]
            or query.shape[0] % keys.shape[1]):
        raise ValueError("incompatible GQA tensors")
    group = query.shape[0] // keys.shape[1]
    head_map = np.arange(query.shape[0]) // group
    scores = np.empty((query.shape[0], keys.shape[0]), dtype=np.float32)
    for bank in range(min(banks, keys.shape[0])):
        positions = np.arange(bank, keys.shape[0], banks)
        scores[:, positions] = np.einsum("hd,nhd->hn", query, keys[positions][:, head_map])
    scores /= math.sqrt(query.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    output = np.zeros_like(query)
    for bank in range(min(banks, keys.shape[0])):
        positions = np.arange(bank, keys.shape[0], banks)
        output += np.einsum("hn,nhd->hd", probabilities[:, positions], values[positions][:, head_map])
    return output


def bank_int8_topk(resident: np.ndarray, query: np.ndarray, *, banks: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    resident = np.asarray(resident)
    query = np.asarray(query)
    if (banks <= 0 or top_k <= 0 or resident.ndim != 2 or query.ndim != 1
            or resident.shape[1] != query.shape[0] or resident.shape[0] == 0
            or resident.dtype != np.int8 or query.dtype != np.int8):
        raise ValueError("valid int8 scan and positive top-k/banks required")
    if resident.shape[1] * 128**2 > np.iinfo(np.int32).max:
        raise ValueError("int32 accumulation could overflow")
    candidates = []
    for bank in range(min(banks, len(resident))):
        positions = np.arange(bank, len(resident), banks)
        scores = resident[positions].astype(np.int32) @ query.astype(np.int32)
        order = np.lexsort((positions, -scores))[:top_k]
        candidates.extend((int(scores[index]), int(positions[index])) for index in order)
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
    return (np.array([item[1] for item in selected], dtype=np.int32),
            np.array([item[0] for item in selected], dtype=np.int32))
