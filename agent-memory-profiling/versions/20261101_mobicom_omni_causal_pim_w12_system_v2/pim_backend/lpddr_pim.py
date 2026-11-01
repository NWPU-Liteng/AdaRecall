"""Shared-LPDDR timing and resident embedding PIM simulation.

This is an architecture simulator, not a phone/Jetson silicon model.  The
host and PIM routes share channels/ranks/banks and therefore cannot claim
independent-DPU speedups.  Every result carries a claim boundary and exposes
the components needed for a trace-replay attribution matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LpddrConfig:
    channels: int = 2
    ranks_per_channel: int = 1
    banks_per_rank: int = 8
    row_bytes: int = 2048
    t_rcd: int = 18
    t_cas: int = 18
    t_rp: int = 18
    t_burst: int = 4
    transfer_bytes: int = 64
    clock_hz: float = 1_000_000_000.0
    refresh_interval: int = 7800
    refresh_cycles: int = 350
    pim_mac_per_cycle: int = 16
    host_prefill_bytes_per_cycle: int = 32

    def __post_init__(self) -> None:
        if min(self.channels, self.ranks_per_channel, self.banks_per_rank, self.row_bytes, self.refresh_interval, self.pim_mac_per_cycle, self.host_prefill_bytes_per_cycle, self.transfer_bytes, self.t_burst, self.clock_hz) <= 0:
            raise ValueError("LPDDR dimensions and timing values must be positive")
        if min(self.t_rcd, self.t_cas, self.t_rp, self.refresh_cycles) < 0:
            raise ValueError("LPDDR timing values must be non-negative")


@dataclass(frozen=True)
class LpddrTraceEvent:
    cycle: int
    actor: str
    channel: int
    rank: int
    bank: int
    row: int
    bytes: int
    kind: str


@dataclass(frozen=True)
class ScanResult:
    route: str
    cycles: int
    bytes_read: int
    score_return_bytes: int
    row_activations: int
    row_hits: int
    precharges: int
    refresh_stalls: int
    bank_conflicts: int
    channel_conflicts: int
    host_sync_cycles: int
    thermal_penalty_cycles: int
    top_indices: tuple[int, ...]
    topk_overlap: float
    trace: tuple[LpddrTraceEvent, ...]
    claim_boundary: str = "shared-LPDDR architecture simulation; not mobile silicon wall-clock"


@dataclass(frozen=True)
class RouterDecision:
    route: str
    predicted_cycles: int
    predicted_energy_pj: float
    thermal_penalty_cycles: int
    alternatives: tuple[tuple[str, int], ...]
    claim_boundary: str = "trace-calibrated route prediction; PIM remains simulated"


class SharedLpddrTimingModel:
    """Deterministic bank/channel timing model with a simple thermal state."""

    def __init__(self, config: LpddrConfig | None = None) -> None:
        self.config = config or LpddrConfig()

    def _location(self, address: int) -> tuple[int, int, int, int]:
        cfg = self.config
        line = address // cfg.transfer_bytes
        bank_total = cfg.channels * cfg.ranks_per_channel * cfg.banks_per_rank
        bank_global = line % bank_total
        channel = bank_global % cfg.channels
        rank_bank = bank_global // cfg.channels
        rank = rank_bank // cfg.banks_per_rank
        bank = rank_bank % cfg.banks_per_rank
        row = (line // bank_total) * cfg.transfer_bytes // cfg.row_bytes
        return channel, rank, bank, row

    def simulate_scan(
        self,
        vector_count: int,
        dimension: int,
        *,
        bytes_per_element: int = 1,
        route: str = "pim",
        host_prefill_bytes: int = 0,
        host_concurrent: bool = True,
        top_indices: Sequence[int] = (),
        reference_top_indices: Sequence[int] = (),
        thermal_state: float = 0.0,
    ) -> ScanResult:
        if vector_count <= 0 or dimension <= 0 or bytes_per_element <= 0:
            raise ValueError("vector_count, dimension, and bytes_per_element must be positive")
        if route not in {"cpu", "gpu", "pim"}:
            raise ValueError("route must be cpu, gpu, or pim")
        if host_prefill_bytes < 0:
            raise ValueError("host_prefill_bytes must be non-negative")
        cfg = self.config
        total_bytes = int(vector_count * dimension * bytes_per_element)
        cycle = 0
        row_open: dict[tuple[int, int, int], int] = {}
        bank_ready: dict[tuple[int, int, int], int] = {}
        channel_ready: dict[int, int] = {}
        row_activations = row_hits = precharges = refresh_stalls = bank_conflicts = channel_conflicts = 0
        sync = 0
        trace: list[LpddrTraceEvent] = []
        next_refresh = cfg.refresh_interval

        def refresh_delay(start: int, duration: int) -> int:
            nonlocal next_refresh, refresh_stalls
            if not cfg.refresh_cycles:
                return start
            while start + duration >= next_refresh:
                start = max(start, next_refresh) + cfg.refresh_cycles
                next_refresh = start + cfg.refresh_interval
                refresh_stalls += 1
                for resource in bank_ready:
                    bank_ready[resource] = max(bank_ready[resource], start)
                for resource in channel_ready:
                    channel_ready[resource] = max(channel_ready[resource], start)
            return start

        if host_concurrent and host_prefill_bytes:
            sync = math.ceil(host_prefill_bytes / cfg.host_prefill_bytes_per_cycle)
            remaining = sync
            while remaining:
                duration = min(remaining, cfg.refresh_interval - 1)
                cycle = refresh_delay(cycle, duration) + duration
                remaining -= duration
        host_finished = cycle
        if route == "cpu":
            cycles_per_byte = 1.0 / max(1, cfg.host_prefill_bytes_per_cycle // 2)
        elif route == "gpu":
            cycles_per_byte = 1.0 / max(1, cfg.host_prefill_bytes_per_cycle)
        else:
            cycles_per_byte = 1.0 / max(1, cfg.pim_mac_per_cycle)
        issue_cycle = host_finished
        for address in range(0, total_bytes, cfg.transfer_bytes):
            channel, rank, bank, row = self._location(address)
            key = (channel, rank, bank)
            read_bytes = min(cfg.transfer_bytes, total_bytes - address)
            bank_wait = bank_ready.get(key, host_finished)
            channel_wait = channel_ready.get(channel, host_finished)
            bank_conflicts += int(bank_wait > issue_cycle)
            channel_conflicts += int(channel_wait > issue_cycle)
            start = max(issue_cycle, bank_wait)
            row_latency = 0
            if row_open.get(key) == row:
                row_hits += 1
            else:
                if key in row_open:
                    precharges += 1
                    row_latency += cfg.t_rp
                row_activations += 1
                row_latency += cfg.t_rcd
                row_open[key] = row
            transfer_start = max(start + row_latency + cfg.t_cas, channel_wait)
            duration = max(cfg.t_burst, math.ceil(read_bytes * cycles_per_byte))
            transfer_start = refresh_delay(transfer_start, duration)
            finish = transfer_start + duration
            bank_ready[key] = finish
            channel_ready[channel] = finish
            cycle = max(cycle, finish)
            issue_cycle = max(issue_cycle + 1, transfer_start - row_latency - cfg.t_cas)
            trace.append(LpddrTraceEvent(finish, route, channel, rank, bank, row, read_bytes, "score_scan"))
        score_return = len(tuple(top_indices)) * 4
        return_cycles = math.ceil(score_return / cfg.host_prefill_bytes_per_cycle)
        sync += return_cycles
        cycle = refresh_delay(cycle, return_cycles) + return_cycles
        thermal = max(0.0, float(thermal_state) + total_bytes / 5_000_000.0 + cycle / 1_000_000.0)
        thermal_penalty = int(max(0.0, thermal - 0.7) * cycle * 0.5)
        cycle += thermal_penalty
        overlap = None
        if len(reference_top_indices):
            overlap = len(set(top_indices) & set(reference_top_indices)) / max(1, len(set(reference_top_indices)))
        else:
            overlap = 1.0
        return ScanResult(route, int(cycle), total_bytes, score_return, row_activations, row_hits, precharges, refresh_stalls, bank_conflicts, channel_conflicts, sync, thermal_penalty, tuple(int(x) for x in top_indices), float(overlap), tuple(trace))


def resident_int8_score_scan(resident: np.ndarray, query: np.ndarray, *, top_k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Numerically exact INT8 dot-product scan used by the PIM functional path."""
    values = np.asarray(resident)
    q = np.asarray(query)
    if values.ndim != 2 or q.ndim != 1 or values.shape[1] != q.shape[0]:
        raise ValueError("resident/query shapes are incompatible")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if values.dtype != np.int8 or q.dtype != np.int8:
        raise ValueError("resident and query must use int8 for the resident path")
    scores = values.astype(np.int32) @ q.astype(np.int32)
    k = min(top_k, len(scores))
    indices = np.argpartition(-scores, k - 1)[:k]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return indices.astype(np.int32), scores[indices].astype(np.int32)


def choose_route(model: SharedLpddrTimingModel, *, vector_count: int, dimension: int, top_k: int, host_prefill_bytes: int, thermal_state: float = 0.0) -> RouterDecision:
    alternatives: list[tuple[str, int]] = []
    for route in ("cpu", "gpu", "pim"):
        result = model.simulate_scan(vector_count, dimension, route=route, host_prefill_bytes=host_prefill_bytes, top_indices=tuple(range(top_k)), thermal_state=thermal_state)
        alternatives.append((route, result.cycles))
    route, cycles = min(alternatives, key=lambda item: (item[1], item[0]))
    return RouterDecision(route, cycles, float(cycles * 12.5), next((int(max(0, cycles - base)) for name, base in alternatives if name == route), 0), tuple(alternatives))
