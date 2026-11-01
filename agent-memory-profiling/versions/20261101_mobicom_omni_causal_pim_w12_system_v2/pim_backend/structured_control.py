"""Bank-resident structured decoding control and one-copy weight residency model."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StructuredControlConfig:
    stacks: int = 2
    channels_per_stack: int = 8
    banks_per_channel: int = 16
    clock_hz: float = 800_000_000.0
    bank_internal_bytes_per_cycle: int = 16
    token_descriptor_bytes: int = 16
    token_test_cycles: int = 64
    reduction_cycles: int = 1024
    command_fence_cycles: int = 2048
    capacity_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        integer_fields = (
            self.stacks, self.channels_per_stack, self.banks_per_channel,
            self.bank_internal_bytes_per_cycle, self.token_descriptor_bytes,
            self.token_test_cycles, self.reduction_cycles,
            self.command_fence_cycles, self.capacity_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in integer_fields):
            raise ValueError("PIM dimensions, costs and capacity must be positive integers")
        if not math.isfinite(self.clock_hz) or self.clock_hz <= 0:
            raise ValueError("PIM clock must be finite and positive")

    @property
    def bank_count(self) -> int:
        return self.stacks * self.channels_per_stack * self.banks_per_channel

    @property
    def bytes_per_cycle(self) -> int:
        return self.bank_count * self.bank_internal_bytes_per_cycle


def mask_scan_cost(token_count: int, vocab_size: int,
                   config: StructuredControlConfig | None = None) -> dict:
    """Estimate a vocabulary-wide grammar scan performed by all PIM banks.

    Every bank owns a disjoint token shard and a read-only copy of the current
    grammar frontier.  The estimate is deliberately pessimistic: each token test
    is charged the configured scalar cycles even though it is a branch-free table
    lookup, and the bitmask is assembled through a fixed cross-bank reduction.
    """
    config = config or StructuredControlConfig()
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("a positive integer token count is required")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("a positive integer vocabulary size is required")
    tokens_per_bank = math.ceil(vocab_size / config.bank_count)
    descriptor_cycles = math.ceil(
        tokens_per_bank * config.token_descriptor_bytes
        / config.bank_internal_bytes_per_cycle)
    test_cycles = tokens_per_bank * config.token_test_cycles
    per_token_cycles = max(descriptor_cycles, test_cycles) + config.reduction_cycles
    total_cycles = token_count * per_token_cycles + config.command_fence_cycles
    return {
        "tokens": token_count,
        "vocab_size": vocab_size,
        "banks": config.bank_count,
        "tokens_per_bank": tokens_per_bank,
        "descriptor_cycles_per_bank_per_token": descriptor_cycles,
        "test_cycles_per_bank_per_token": test_cycles,
        "reduction_cycles_per_token": config.reduction_cycles,
        "command_fence_cycles": config.command_fence_cycles,
        "cycles": total_cycles,
        "seconds": total_cycles / config.clock_hz,
        "milliseconds_per_token": 1000 * per_token_cycles / config.clock_hz,
        "descriptor_bytes_per_token": config.bank_count * tokens_per_bank
        * config.token_descriptor_bytes,
        "boundary": ("analytical bank-parallel grammar scan; no RTL, silicon or "
                     "cycle-accurate command/refresh simulation"),
    }


def residency_layout(weight_bytes: int, *, token_table_bytes: int,
                     grammar_state_bytes: int, scratch_bytes: int,
                     config: StructuredControlConfig | None = None) -> dict:
    """Check one-copy model, token-table, control-state and scratch residency."""
    config = config or StructuredControlConfig()
    values = (weight_bytes, token_table_bytes, grammar_state_bytes, scratch_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise ValueError("residency byte counts must be nonnegative integers")
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
        "boundary": "static capacity accounting; refresh, repair, metadata and OS reservation included only through scratch_bytes",
    }


def project_control_plane(measured_host_seconds: float, token_count: int,
                          vocab_size: int,
                          config: StructuredControlConfig | None = None) -> dict:
    """Compare a measured host grammar replay with the bank-parallel model."""
    if (not math.isfinite(measured_host_seconds) or measured_host_seconds <= 0):
        raise ValueError("positive finite measured host grammar time required")
    modeled = mask_scan_cost(token_count, vocab_size, config)
    return {
        "measured_host_grammar_seconds": measured_host_seconds,
        "modeled_pim_grammar_seconds": modeled["seconds"],
        "modeled_control_speedup": measured_host_seconds / modeled["seconds"],
        "modeled_pim_grammar_milliseconds_per_token": modeled["milliseconds_per_token"],
        "model": modeled,
        "boundary": ("comparison of a measured CPU replay with an analytical bank-parallel "
                     "model; not a measured PIM device result"),
    }
