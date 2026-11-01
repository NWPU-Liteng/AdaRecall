"""Unified Qwen3.5 state-tier PIM planning and reference primitives.

The active design treats the native hybrid state as one capacity object.  PIM
may own both full-attention KV and linear-attention recurrent/conv state, and
executes only bank-local state accesses: GQA QK/AV for KV and the recurrent
update for linear state.  Projections, softmax, cross-bank reduction, MLPs,
nonlinearities and semantic Reader work remain Host/GPU-owned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .lpddr_bank_pim import BankPimConfig, BankPimModel


SCHEMA = "hybrid-state-pim-plan-v2"
FUNCTIONAL_SCHEMA = "unified-hybrid-state-reference-v2"


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class HybridStateLayout:
    """Exact resident-state layout extracted from a native cache inventory."""

    linear_layers: int
    recurrent_state_bytes: int
    conv_state_bytes: int
    full_attention_bytes: int = 0
    recurrent_heads: int = 32
    key_dim: int = 128
    value_dim: int = 128
    recurrent_element_bytes: int = 4
    conv_element_bytes: int = 2
    full_attention_layers: int = 8
    full_attention_context_tokens: int = 24000
    full_attention_query_heads: int = 16
    full_attention_kv_heads: int = 4
    full_attention_head_dim: int = 256
    full_attention_element_bytes: int = 2

    def __post_init__(self) -> None:
        for field in (
            "linear_layers", "recurrent_state_bytes", "conv_state_bytes",
            "recurrent_heads", "key_dim", "value_dim", "recurrent_element_bytes",
            "conv_element_bytes", "full_attention_element_bytes",
            "full_attention_query_heads",
        ):
            _positive_int(getattr(self, field), field)
        if self.full_attention_bytes < 0:
            raise ValueError("full_attention_bytes must be nonnegative")
        full_fields = (
            self.full_attention_layers,
            self.full_attention_context_tokens,
            self.full_attention_kv_heads,
            self.full_attention_head_dim,
        )
        if self.full_attention_bytes and any(value <= 0 for value in full_fields):
            raise ValueError(
                "full-attention metadata is required when full_attention_bytes is nonzero"
            )

    @property
    def linear_state_bytes(self) -> int:
        return self.recurrent_state_bytes + self.conv_state_bytes

    @property
    def total_native_state_bytes(self) -> int:
        return self.linear_state_bytes + self.full_attention_bytes

    @property
    def full_hybrid_state_bytes(self) -> int:
        """The complete model-native state that a unified tier must retain."""
        return self.total_native_state_bytes

    @classmethod
    def from_cache_inventory(
        cls,
        inventory: dict[str, Any],
        *,
        full_attention_query_heads: int = 16,
    ) -> "HybridStateLayout":
        if not isinstance(inventory, dict):
            raise ValueError("cache inventory object required")
        layers = inventory.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("cache inventory layers required")
        linear = [item for item in layers if item.get("kind") == "linear_attention"]
        full = [item for item in layers if item.get("kind") == "full_attention"]
        if not linear:
            raise ValueError("hybrid layout requires linear-attention layers")
        recurrent_bytes = 0
        conv_bytes = 0
        recurrent_shape = None
        recurrent_dtype_bytes = None
        conv_dtype_bytes = None
        full_shape = None
        full_dtype_bytes = None
        for item in linear:
            shapes = item.get("tensor_shapes", [])
            dtypes = item.get("dtypes", [])
            if len(shapes) < 2:
                raise ValueError("linear-attention layer must expose conv and recurrent shapes")
            conv_shape, state_shape = shapes[0], shapes[1]
            if len(conv_shape) != 3 or len(state_shape) != 4:
                raise ValueError("unsupported native hybrid state shape")
            state_bytes = int(np.prod(state_shape)) * _dtype_bytes(dtypes, preferred="torch.float32")
            layer_conv_bytes = int(np.prod(conv_shape)) * _dtype_bytes(
                dtypes, preferred="torch.bfloat16")
            recurrent_bytes += state_bytes
            conv_bytes += layer_conv_bytes
            recurrent_shape = state_shape
            recurrent_dtype_bytes = state_bytes // int(np.prod(state_shape))
            conv_dtype_bytes = layer_conv_bytes // int(np.prod(conv_shape))
        if full:
            full_shapes = full[0].get("tensor_shapes", [])
            full_dtypes = full[0].get("dtypes", [])
            if not full_shapes or len(full_shapes[0]) != 4:
                raise ValueError("full-attention layer must expose KV shape")
            full_shape = full_shapes[0]
            full_dtype_bytes = _dtype_bytes(full_dtypes, preferred="torch.bfloat16")
        if recurrent_shape is None:
            raise ValueError("recurrent state shape missing")
        return cls(
            linear_layers=len(linear),
            recurrent_state_bytes=recurrent_bytes,
            conv_state_bytes=conv_bytes,
            full_attention_bytes=int(inventory.get("full_attention_bytes", 0)),
            recurrent_heads=int(recurrent_shape[1]),
            key_dim=int(recurrent_shape[2]),
            value_dim=int(recurrent_shape[3]),
            recurrent_element_bytes=int(recurrent_dtype_bytes or 4),
            conv_element_bytes=int(conv_dtype_bytes or 2),
            full_attention_layers=len(full),
            full_attention_context_tokens=int(full_shape[2]) if full_shape else 0,
            full_attention_query_heads=full_attention_query_heads,
            full_attention_kv_heads=int(full_shape[1]) if full_shape else 0,
            full_attention_head_dim=int(full_shape[3]) if full_shape else 0,
            full_attention_element_bytes=int(full_dtype_bytes or 2),
        )


def _dtype_bytes(dtypes: list[str], *, preferred: str) -> int:
    if preferred in dtypes:
        return 4 if preferred.endswith("float32") else 2
    if any(str(item).endswith("float32") for item in dtypes):
        return 4
    if any(str(item).endswith(("float16", "bfloat16")) for item in dtypes):
        return 2
    raise ValueError("unsupported state dtype")


def bank_recurrent_state_step(
    state: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    decay: np.ndarray,
    *,
    banks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference bank-sharded recurrent update with exact FP32 accumulation.

    The primitive is intentionally generic: it captures the state-resident
    outer-product/update and query-state read used by linear attention. It is
    not a drop-in replacement for the opaque Qwen3.5 kernel until a model
    specific numerical contract is separately proven.
    """
    state = np.asarray(state, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    decay = np.asarray(decay, dtype=np.float32)
    if banks <= 0 or state.ndim != 3 or query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("invalid recurrent state tensors")
    if (state.shape[0] != query.shape[0] or state.shape[0] != key.shape[0]
            or state.shape[0] != value.shape[0]):
        raise ValueError("head dimensions do not agree")
    if state.shape[1] != key.shape[1] or state.shape[2] != value.shape[1]:
        raise ValueError("state and update dimensions do not agree")
    if query.shape[1] != state.shape[1] or decay.shape not in {(state.shape[0],), (state.shape[0], 1)}:
        raise ValueError("query or decay dimensions do not agree")
    decay = decay.reshape(-1)
    updated = np.empty_like(state)
    output = np.empty_like(value)
    for bank in range(min(banks, state.shape[0])):
        heads = np.arange(bank, state.shape[0], banks)
        updated[heads] = decay[heads, None, None] * state[heads]
        updated[heads] += key[heads, :, None] * value[heads, None, :]
        output[heads] = np.einsum("hk,hkv->hv", query[heads], updated[heads])
    return updated, output


def bank_gated_delta_state_step(
    state: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gate_log_decay: np.ndarray,
    beta: np.ndarray,
    *,
    banks: int,
    norm_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3.5 single-token Gated Delta Rule reference with bank ownership.

    Inputs correspond to the post-convolution, post-projection per-head values
    consumed by the recurrent rule. The state update and query-state read remain
    within the assigned bank; only the resulting attention output is returned.
    This NumPy reference checks semantics and does not execute PIM hardware.
    """
    state = np.asarray(state, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    key = np.asarray(key, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    gate_log_decay = np.asarray(gate_log_decay, dtype=np.float32).reshape(-1)
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    if (
        banks <= 0
        or state.ndim != 3
        or query.ndim != 2
        or key.shape != query.shape
        or value.ndim != 2
        or state.shape[0] != query.shape[0]
        or state.shape[0] != value.shape[0]
        or state.shape[1] != query.shape[1]
        or state.shape[2] != value.shape[1]
        or gate_log_decay.shape != (state.shape[0],)
        or beta.shape != (state.shape[0],)
        or norm_epsilon <= 0
    ):
        raise ValueError("incompatible Gated Delta Rule tensors")

    updated = np.empty_like(state)
    output = np.empty_like(value)
    for bank in range(min(banks, state.shape[0])):
        heads = np.arange(bank, state.shape[0], banks)
        q = query[heads]
        k = key[heads]
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + norm_epsilon)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + norm_epsilon)
        q = q / math.sqrt(state.shape[1])
        decayed = state[heads] * np.exp(gate_log_decay[heads, None, None])
        key_memory = np.sum(decayed * k[:, :, None], axis=1)
        delta = (value[heads] - key_memory) * beta[heads, None]
        updated[heads] = decayed + k[:, :, None] * delta[:, None, :]
        output[heads] = np.sum(updated[heads] * q[:, :, None], axis=1)
    return updated, output


def bank_online_softmax_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    banks: int,
) -> np.ndarray:
    """Exact GQA attention reference using bank-local online-softmax summaries.

    Each bank consumes only its strided KV shard and emits (max, denominator,
    weighted-value) summaries. The summaries merge associatively, avoiding a
    global score-vector materialization and a second KV read in the reference
    algorithm. This is a functional mapping reference, not a physical PIM run.
    """
    query = np.asarray(query, dtype=np.float32)
    keys = np.asarray(keys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if (
        banks <= 0
        or query.ndim != 2
        or keys.ndim != 3
        or values.shape != keys.shape
        or min(query.shape) <= 0
        or min(keys.shape) <= 0
        or query.shape[1] != keys.shape[2]
        or query.shape[0] % keys.shape[1]
    ):
        raise ValueError("incompatible GQA tensors")

    query_heads, head_dim = query.shape
    kv_heads = keys.shape[1]
    group_size = query_heads // kv_heads
    head_map = np.arange(query_heads) // group_size
    scale = 1.0 / math.sqrt(head_dim)
    summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for bank in range(min(banks, keys.shape[0])):
        positions = np.arange(bank, keys.shape[0], banks)
        bank_keys = keys[positions][:, head_map]
        bank_values = values[positions][:, head_map]
        scores = np.einsum("hd,nhd->hn", query, bank_keys) * scale
        local_max = scores.max(axis=1)
        weights = np.exp(scores - local_max[:, None])
        local_sum = weights.sum(axis=1)
        local_output = np.einsum("hn,nhd->hd", weights, bank_values)
        summaries.append((local_max, local_sum, local_output))

    global_max = np.maximum.reduce([item[0] for item in summaries])
    global_sum = np.zeros(query_heads, dtype=np.float32)
    global_output = np.zeros((query_heads, head_dim), dtype=np.float32)
    for local_max, local_sum, local_output in summaries:
        rescale = np.exp(local_max - global_max)
        global_sum += local_sum * rescale
        global_output += local_output * rescale[:, None]
    return global_output / global_sum[:, None]


@dataclass(frozen=True)
class HybridStatePimPlanner:
    config: BankPimConfig
    reserved_bytes: int = 0
    input_bytes_per_layer: int = 24576
    output_bytes_per_layer: int = 8192
    accumulator_bytes: int = 4

    def __post_init__(self) -> None:
        if self.reserved_bytes < 0 or self.input_bytes_per_layer <= 0:
            raise ValueError("invalid reserved or input bytes")
        if self.output_bytes_per_layer <= 0 or self.accumulator_bytes <= 0:
            raise ValueError("invalid output or accumulator bytes")
        if self.reserved_bytes >= self.config.pim_capacity_bytes:
            raise ValueError("reserved bytes consume the entire PIM pool")

    def admission(
        self,
        layout: HybridStateLayout,
        *,
        active_agents: int,
        state_scope: str = "linear",
    ) -> dict[str, Any]:
        if state_scope not in {"linear", "unified"}:
            raise ValueError("state_scope must be linear or unified")
        _positive_int(active_agents, "active_agents")
        per_agent = (
            layout.linear_state_bytes
            if state_scope == "linear" else layout.full_hybrid_state_bytes
        )
        available = self.config.pim_capacity_bytes - self.reserved_bytes
        resident_agents = available // per_agent
        requested = per_agent * active_agents
        reasons = []
        if requested > available:
            reasons.append("active_agent_capacity")
        per_bank_workspace = math.ceil(
            (self.input_bytes_per_layer + self.output_bytes_per_layer)
            / self.config.bank_count
        ) + self.accumulator_bytes * layout.recurrent_heads
        if per_bank_workspace > self.config.local_buffer_bytes_per_bank:
            reasons.append("bank_workspace")
        return {
            "admitted": not reasons,
            "reasons": reasons,
            "state_scope": state_scope,
            "active_agents": active_agents,
            "resident_agents": int(resident_agents),
            "per_agent_state_bytes": per_agent,
            "requested_state_bytes": requested,
            "available_bytes": available,
            "per_agent_linear_state_bytes": layout.linear_state_bytes,
            "per_agent_full_attention_bytes": layout.full_attention_bytes,
            "requested_linear_state_bytes": layout.linear_state_bytes * active_agents,
            "requested_full_attention_bytes": layout.full_attention_bytes * active_agents,
            "full_attention_bytes_not_in_pim": (
                layout.full_attention_bytes if state_scope == "linear" else 0
            ),
            "bank_workspace_bytes": per_bank_workspace,
        }

    def step_cost(self, layout: HybridStateLayout, *, route: str = "pim") -> dict[str, Any]:
        if route not in {"gpu_resident", "host_offloaded", "pim"}:
            raise ValueError("route must be gpu_resident, host_offloaded or pim")
        cfg = self.config
        state_bytes = layout.linear_state_bytes
        layers = layout.linear_layers
        state_macs = layers * layout.recurrent_heads * layout.key_dim * layout.value_dim
        read_macs = state_macs
        total_macs = state_macs + read_macs
        io_bytes = layers * (self.input_bytes_per_layer + self.output_bytes_per_layer)
        state_memory_cycles = math.ceil(
            state_bytes / (cfg.bank_count * cfg.bank_internal_bytes_per_cycle)
        )
        pim_compute_cycles = math.ceil(
            total_macs / (cfg.bank_count * cfg.pim_macs_per_bank_cycle)
        )
        host_compute_cycles = math.ceil(total_macs / cfg.host_macs_per_cycle)
        if route == "gpu_resident":
            active_cycles = cfg.host_command_cycles + host_compute_cycles + cfg.host_fence_cycles
            external_bytes = io_bytes
            internal_bytes = 0
            commands = 1
            fences = 1
            phases = [{
                "name": "gpu_resident_state_update",
                "active_cycles": active_cycles,
                "external_bytes": external_bytes,
                "bank_internal_bytes": 0,
                "macs": total_macs,
            }]
        elif route == "host_offloaded":
            external_bytes = 2 * state_bytes + io_bytes
            active_cycles = cfg.host_command_cycles + max(
                self._backend().transfer_cycles(external_bytes), host_compute_cycles
            ) + cfg.host_fence_cycles
            internal_bytes = 0
            commands = 1
            fences = 1
            phases = [{
                "name": "host_state_roundtrip_and_update",
                "active_cycles": active_cycles,
                "external_bytes": external_bytes,
                "bank_internal_bytes": 0,
                "macs": total_macs,
            }]
        else:
            submit_cycles = cfg.host_command_cycles + self._backend().transfer_cycles(
                layers * self.input_bytes_per_layer
            )
            update_cycles = state_memory_cycles + pim_compute_cycles
            return_cycles = self._backend().transfer_cycles(layers * self.output_bytes_per_layer)
            active_cycles = submit_cycles + update_cycles + return_cycles + cfg.host_fence_cycles
            external_bytes = io_bytes
            internal_bytes = 2 * state_bytes
            commands = 2
            fences = 1
            phases = [
                {
                    "name": "pim_state_update",
                    "active_cycles": update_cycles,
                    "external_bytes": 0,
                    "bank_internal_bytes": internal_bytes,
                    "macs": total_macs,
                },
                {
                    "name": "pim_host_boundary",
                    "active_cycles": submit_cycles + return_cycles + cfg.host_fence_cycles,
                    "external_bytes": external_bytes,
                    "bank_internal_bytes": 0,
                    "macs": 0,
                },
            ]
        cycles = self._backend().refresh_elapsed(active_cycles)
        return {
            "schema": SCHEMA,
            "route": route,
            "cycles": cycles,
            "active_cycles": active_cycles,
            "refresh_cycles": cycles - active_cycles,
            "external_bytes": external_bytes,
            "bank_internal_bytes": internal_bytes,
            "state_bytes": state_bytes,
            "io_bytes": io_bytes,
            "macs": total_macs,
            "host_commands": commands,
            "host_fences": fences,
            "phases": phases,
            "full_attention_untouched": True,
            "boundary": (
                "hybrid linear-attention recurrent/conv state transition only; "
                "full-attention KV, MLP, projections, nonlinearities and Reader work remain Host"
            ),
        }

    def full_attention_step_cost(
        self, layout: HybridStateLayout, *, route: str
    ) -> dict[str, Any]:
        """Model bank-local GQA over resident full-attention KV.

        The method intentionally keeps softmax and cross-bank reduction on the
        Host, matching the LPDDR bank-PIM profile instead of pretending that a
        complete attention operator runs inside memory.
        """
        if route not in {"host", "pim"}:
            raise ValueError("full-attention route must be host or pim")
        if not layout.full_attention_bytes:
            return {
                "route": route,
                "cycles": 0,
                "external_bytes": 0,
                "bank_internal_bytes": 0,
                "macs": 0,
                "phases": [],
                "boundary": "no full-attention KV in layout",
            }
        per_layer = self._backend().attention_step(
            context_tokens=layout.full_attention_context_tokens,
            query_heads=layout.full_attention_query_heads,
            kv_heads=layout.full_attention_kv_heads,
            head_dim=layout.full_attention_head_dim,
            element_bytes=layout.full_attention_element_bytes,
            route=route,
        )
        return {
            **per_layer,
            "cycles_per_layer": per_layer["cycles"],
            "cycles": per_layer["cycles"] * layout.full_attention_layers,
            "external_bytes": per_layer["external_bytes"] * layout.full_attention_layers,
            "bank_internal_bytes": per_layer["bank_internal_bytes"] * layout.full_attention_layers,
            "macs": per_layer["macs"] * layout.full_attention_layers,
            "row_activations": per_layer["row_activations"] * layout.full_attention_layers,
            "host_commands": per_layer["host_commands"] * layout.full_attention_layers,
            "host_fences": per_layer["host_fences"] * layout.full_attention_layers,
            "full_attention_layers": layout.full_attention_layers,
            "boundary": (
                "full-attention KV resident in LPDDR banks; QK/AV are bank-local, "
                "while softmax and cross-bank reduction remain Host-owned"
            ),
        }

    def unified_step_cost(self, layout: HybridStateLayout, *, route: str) -> dict[str, Any]:
        """Cost the complete native state tier rather than only linear state."""
        if route not in {"host_offloaded", "pim"}:
            raise ValueError("unified route must be host_offloaded or pim")
        linear_route = route
        full_route = "host" if route == "host_offloaded" else "pim"
        linear = self.step_cost(layout, route=linear_route)
        full = self.full_attention_step_cost(layout, route=full_route)
        phases = [
            {"name": "linear_state", "cycles": linear["cycles"],
             "external_bytes": linear["external_bytes"],
             "bank_internal_bytes": linear["bank_internal_bytes"]},
            {"name": "full_attention_kv", "cycles": full["cycles"],
             "external_bytes": full["external_bytes"],
             "bank_internal_bytes": full["bank_internal_bytes"]},
        ]
        cycles = linear["cycles"] + full["cycles"]
        return {
            "schema": SCHEMA,
            "route": route,
            "cycles": cycles,
            "external_bytes": linear["external_bytes"] + full["external_bytes"],
            "bank_internal_bytes": linear["bank_internal_bytes"] + full["bank_internal_bytes"],
            "state_bytes": layout.full_hybrid_state_bytes,
            "linear_state_bytes": layout.linear_state_bytes,
            "full_attention_bytes": layout.full_attention_bytes,
            "host_commands": linear["host_commands"] + full.get("host_commands", 0),
            "host_fences": linear["host_fences"] + full.get("host_fences", 0),
            "phases": phases,
            "full_attention_managed_by_pim": route == "pim",
            "boundary": (
                "unified native-state tier: full-attention KV and linear recurrent/conv "
                "state are resident in the selected tier; model projections, softmax, "
                "cross-bank reduction, MLPs and Reader remain Host/GPU-owned"
            ),
        }

    def unified_plan(self, layout: HybridStateLayout, *, active_agents: int) -> dict[str, Any]:
        admission = self.admission(layout, active_agents=active_agents, state_scope="unified")
        host = self.unified_step_cost(layout, route="host_offloaded")
        pim = self.unified_step_cost(layout, route="pim")
        available = admission["available_bytes"]
        required = admission["requested_state_bytes"]
        return {
            "schema": SCHEMA,
            "state_scope": "unified_hybrid_state",
            "admission": admission,
            "host_offloaded_step": host,
            "pim_step": pim,
            "conditional_modeled_step_speedup_for_admitted_resident_state": (
                host["cycles"] / pim["cycles"]
                if admission["admitted"] and pim["cycles"] else None),
            "per_state_modeled_step_ratio_diagnostic_not_workload_speedup": (
                host["cycles"] / pim["cycles"] if pim["cycles"] else None),
            "required_pool_bytes_for_workload": required,
            "capacity_scale_factor_required": required / available if available else None,
            "full_attention_managed_by_pim": True,
            "linear_attention_managed_by_pim": True,
            "physical_pim_executed": False,
            "end_to_end_speedup": None,
            "publication_ready": False,
            "claim_boundary": (
                "unified full-attention-KV plus recurrent-state phase model; current "
                "profile must pass the complete-state capacity gate before any workload "
                "speedup can be claimed"
            ),
        }

    def plan(self, layout: HybridStateLayout, *, active_agents: int) -> dict[str, Any]:
        admission = self.admission(layout, active_agents=active_agents, state_scope="linear")
        host = self.step_cost(layout, route="host_offloaded")
        pim = self.step_cost(layout, route="pim")
        return {
            "schema": SCHEMA,
            "admission": admission,
            "gpu_resident_step": self.step_cost(layout, route="gpu_resident"),
            "host_offloaded_step": host,
            "pim_step": pim,
            "full_attention_untouched": True,
            "conditional_modeled_step_speedup_for_admitted_state": (
                host["cycles"] / pim["cycles"]
                if admission["admitted"] and pim["cycles"] else None),
            "per_state_modeled_step_ratio_diagnostic_not_workload_speedup": (
                host["cycles"] / pim["cycles"] if pim["cycles"] else None),
            "modeled_step_speedup_vs_gpu_resident": (
                self.step_cost(layout, route="gpu_resident")["cycles"] / pim["cycles"]
                if pim["cycles"] else None),
            "resident_state_scope": "linear_attention recurrent and conv state only",
            "nonresident_scope": "full-attention KV, all model weights, MLP, projections, nonlinearities, semantic Reader",
            "physical_pim_executed": False,
            "end_to_end_speedup": None,
            "publication_ready": False,
            "claim_boundary": (
                "uncalibrated shared-LPDDR phase model plus functional state reference; "
                "not a physical PIM measurement or current 200-question E2E result"
            ),
        }

    def _backend(self) -> BankPimModel:
        return BankPimModel(self.config)


__all__ = [
    "FUNCTIONAL_SCHEMA",
    "SCHEMA",
    "HybridStateLayout",
    "HybridStatePimPlanner",
    "bank_online_softmax_attention",
    "bank_gated_delta_state_step",
    "bank_recurrent_state_step",
]
