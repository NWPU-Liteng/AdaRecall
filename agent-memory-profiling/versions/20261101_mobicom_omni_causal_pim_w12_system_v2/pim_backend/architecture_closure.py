"""Target-architecture counterfactual for evidence-state materialization.

This module consumes measured server request phases and replaces only the
proof-materialization phase with a transparent, banked-memory PIM model.  It
does not model semantic grounding, retrieval, Reader inference, or physical
device behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from statistics import median

from .lpddr_pim import SharedLpddrTimingModel


@dataclass(frozen=True)
class EvidencePimArchitectureConfig:
    channels: int = 2
    banks_per_channel: int = 8
    row_bytes: int = 2048
    burst_bytes: int = 64
    t_activate: int = 18
    t_read: int = 18
    t_precharge: int = 18
    t_burst: int = 4
    clock_hz: float = 1_000_000_000.0
    pim_bytes_per_cycle: int = 32
    setup_cycles: int = 100_000
    fence_cycles: int = 20_000
    write_amplification: float = 1.0

    def __post_init__(self):
        positive = (self.channels, self.banks_per_channel, self.row_bytes,
                    self.burst_bytes, self.clock_hz, self.pim_bytes_per_cycle)
        if any(value <= 0 for value in positive):
            raise ValueError("architecture dimensions and rates must be positive")
        if min(self.t_activate, self.t_read, self.t_precharge, self.t_burst,
               self.setup_cycles, self.fence_cycles) < 0:
            raise ValueError("timing values must be nonnegative")
        if not math.isfinite(self.write_amplification) or self.write_amplification < 1:
            raise ValueError("write amplification must be finite and at least one")


def _stream_cycles(byte_count: int, config: EvidencePimArchitectureConfig) -> dict:
    if byte_count < 0:
        raise ValueError("byte count must be nonnegative")
    if byte_count == 0:
        return {"cycles": 0, "bursts": 0, "row_activations": 0,
                "row_hits": 0, "bank_conflicts": 0}
    bank_count = config.channels * config.banks_per_channel
    row_open = [None] * bank_count
    bank_ready = [0] * bank_count
    cycle = 0
    row_activations = row_hits = bank_conflicts = 0
    bursts = math.ceil(byte_count / config.burst_bytes)
    for burst_index in range(bursts):
        bank = burst_index % bank_count
        row = (burst_index * config.burst_bytes // bank_count) // config.row_bytes
        start = max(cycle, bank_ready[bank])
        if start > cycle:
            bank_conflicts += 1
        if row_open[bank] == row:
            start += config.t_read
            row_hits += 1
        else:
            if row_open[bank] is not None:
                start += config.t_precharge
            start += config.t_activate + config.t_read
            row_open[bank] = row
            row_activations += 1
        finish = start + config.t_burst
        bank_ready[bank] = finish
        cycle = finish
    return {"cycles": cycle, "bursts": bursts, "row_activations": row_activations,
            "row_hits": row_hits, "bank_conflicts": bank_conflicts}


def model_materialization(payload_bytes: int, metadata_bytes: int,
                          config: EvidencePimArchitectureConfig) -> dict:
    """Model exact state assembly in shared LPDDR without host round-trip."""
    if payload_bytes < 0 or metadata_bytes < 0:
        raise ValueError("payload and metadata bytes must be nonnegative")
    packed_bytes = payload_bytes + metadata_bytes
    input_bytes = payload_bytes + metadata_bytes
    output_bytes = math.ceil(packed_bytes * config.write_amplification)
    traffic_bytes = input_bytes + output_bytes
    memory = _stream_cycles(traffic_bytes, config)
    compute_cycles = math.ceil(traffic_bytes / config.pim_bytes_per_cycle)
    cycles = (memory["cycles"] + compute_cycles + config.setup_cycles
              + config.fence_cycles)
    return {
        "payload_bytes": int(payload_bytes),
        "metadata_bytes": int(metadata_bytes),
        "packed_bytes": int(packed_bytes),
        "traffic_bytes": int(traffic_bytes),
        "memory_cycles": memory["cycles"],
        "compute_cycles": compute_cycles,
        "setup_cycles": config.setup_cycles,
        "fence_cycles": config.fence_cycles,
        "cycles": cycles,
        "seconds": cycles / config.clock_hz,
        "row_activations": memory["row_activations"],
        "row_hits": memory["row_hits"],
        "bank_conflicts": memory["bank_conflicts"],
        "claim_boundary": "target architecture model; no device calibration or physical PIM execution",
    }


def _quantile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires nonempty values")
    return ordered[max(0, min(len(ordered) - 1,
                               math.ceil(quantile * len(ordered)) - 1))]


@lru_cache(maxsize=64)
def _modeled_scan_speedup(vector_count: int) -> float:
    timing = SharedLpddrTimingModel()
    host_scan = timing.simulate_scan(vector_count, 1024, bytes_per_element=4,
                                     route="cpu")
    pim_scan = timing.simulate_scan(vector_count, 1024, bytes_per_element=1,
                                    route="pim")
    return host_scan.cycles / pim_scan.cycles


def project_rows(rows: list[dict], config: EvidencePimArchitectureConfig,
                 *, quality_contract_assumed: bool = True,
                 speedup_gate: float = 1.05) -> dict:
    """Replace only proof materialization and report complete-request metrics."""
    projected = []
    for row in rows:
        phases = row["event_seconds"]
        host_service = float(row["service_seconds"])
        host_ttft = sum(float(phases[name]) for name in (
            "query_parse_and_retrieval", "source_grounding",
            "proof_materialize", "reader_prefill"))
        host_stage = float(phases["proof_materialize"])
        pim = model_materialization(int(row["payload_bytes"]),
                                     int(row["metadata_bytes"]), config)
        candidate_service = host_service - host_stage + pim["seconds"]
        candidate_ttft = host_ttft - host_stage + pim["seconds"]
        projected.append({
            "question_id": row["question_id"],
            "host_service_seconds": host_service,
            "candidate_service_seconds": candidate_service,
            "host_ttft_seconds": host_ttft,
            "candidate_ttft_seconds": candidate_ttft,
            "host_materialize_seconds": host_stage,
            "pim_materialize": pim,
            "capacity_admitted": int(row["packed_bytes"]) <= 2 * 1024 ** 3,
            "quality_contract_assumed": bool(quality_contract_assumed),
        })
    host_service = [item["host_service_seconds"] for item in projected]
    candidate_service = [item["candidate_service_seconds"] for item in projected]
    host_ttft = [item["host_ttft_seconds"] for item in projected]
    candidate_ttft = [item["candidate_ttft_seconds"] for item in projected]
    service_host_total = sum(host_service)
    service_candidate_total = sum(candidate_service)
    ttft_host_total = sum(host_ttft)
    ttft_candidate_total = sum(candidate_ttft)
    service_speedup = service_host_total / service_candidate_total
    ttft_speedup = ttft_host_total / ttft_candidate_total
    p95_host = _quantile(host_service, .95)
    p95_candidate = _quantile(candidate_service, .95)
    return {
        "schema": "evidence-pim-architecture-closure-v1",
        "model_kind": "target_architecture_counterfactual",
        "requests": len(projected),
        "configuration": config.__dict__,
        "quality_contract_assumed": bool(quality_contract_assumed),
        "metrics": {
            "host_service_seconds_total": service_host_total,
            "candidate_service_seconds_total": service_candidate_total,
            "host_ttft_seconds_total": ttft_host_total,
            "candidate_ttft_seconds_total": ttft_candidate_total,
            "service_speedup": service_speedup,
            "ttft_speedup": ttft_speedup,
            "service_p95_host_seconds": p95_host,
            "service_p95_candidate_seconds": p95_candidate,
            "p95_ratio": p95_host / p95_candidate,
            "p95_host_over_candidate": p95_host / p95_candidate,
            "materialize_host_seconds_total": sum(item["host_materialize_seconds"] for item in projected),
            "materialize_pim_seconds_total": sum(item["pim_materialize"]["seconds"] for item in projected),
            "materialize_traffic_bytes_total": sum(item["pim_materialize"]["traffic_bytes"] for item in projected),
            "materialize_host_fraction": sum(item["host_materialize_seconds"] for item in projected) / service_host_total,
        },
        "gates": {
            "quality_contract_assumed": bool(quality_contract_assumed),
            "capacity_all_admitted": all(item["capacity_admitted"] for item in projected),
            "service_speedup_pass": service_speedup >= speedup_gate,
            "ttft_speedup_pass": ttft_speedup >= speedup_gate,
            "p95_not_worse": p95_candidate <= p95_host,
        },
        "positive_architecture_gate_pass": bool(
            quality_contract_assumed and
            all(item["capacity_admitted"] for item in projected) and
            service_speedup >= speedup_gate and ttft_speedup >= speedup_gate and
            p95_candidate <= p95_host),
        "rows": projected,
        "claim_boundary": (
            "server-measured request phases plus explicit target architecture model; "
            "not physical device latency, power, or calibrated mobile PIM speedup"),
    }


def sensitivity(rows: list[dict], base: EvidencePimArchitectureConfig,
                *, speedup_gate: float = 1.05) -> list[dict]:
    results = []
    for rate_scale in (0.25, 1.0, 4.0):
        for overhead_scale in (0.25, 1.0, 4.0):
            config = replace(
                base,
                pim_bytes_per_cycle=max(1, round(base.pim_bytes_per_cycle * rate_scale)),
                setup_cycles=round(base.setup_cycles * overhead_scale),
                fence_cycles=round(base.fence_cycles * overhead_scale),
            )
            result = project_rows(rows, config, speedup_gate=speedup_gate)
            results.append({
                "rate_scale": rate_scale,
                "overhead_scale": overhead_scale,
                "service_speedup": result["metrics"]["service_speedup"],
                "ttft_speedup": result["metrics"]["ttft_speedup"],
                "p95_ratio": result["metrics"]["p95_ratio"],
                "positive_architecture_gate_pass": result["positive_architecture_gate_pass"],
            })
    return results


def project_joint_retrieval_rows(
        rows: list[dict], vector_counts: dict[str, int],
        config: EvidencePimArchitectureConfig, *,
        retrieval_scan_fraction: float,
        quality_contract_assumed: bool = True,
        speedup_gate: float = 1.05) -> dict:
    """Combine modeled resident-vector scan and evidence materialization.

    The scan fraction is intentionally an explicit unknown.  It must not be
    inferred from the aggregate query/retrieval wall time; the output is a
    sensitivity projection until a serving trace separates embedding, index,
    rerank and proof work.
    """
    if not 0 <= retrieval_scan_fraction <= 1:
        raise ValueError("retrieval scan fraction must be in [0, 1]")
    projected = []
    for row in rows:
        person_id = str(row["person_id"])
        vector_count = int(vector_counts[person_id])
        scan_speedup = _modeled_scan_speedup(vector_count)
        phases = row["event_seconds"]
        retrieval = float(phases["query_parse_and_retrieval"])
        materialize_host = float(phases["proof_materialize"])
        host_service = float(row["service_seconds"])
        host_ttft = sum(float(phases[name]) for name in (
            "query_parse_and_retrieval", "source_grounding",
            "proof_materialize", "reader_prefill"))
        materialize = model_materialization(int(row["payload_bytes"]),
                                            int(row["metadata_bytes"]), config)
        candidate_retrieval = retrieval * (
            1 - retrieval_scan_fraction + retrieval_scan_fraction / scan_speedup)
        candidate_service = (host_service - retrieval - materialize_host
                             + candidate_retrieval + materialize["seconds"])
        candidate_ttft = (host_ttft - retrieval - materialize_host
                          + candidate_retrieval + materialize["seconds"])
        projected.append({
            "question_id": row["question_id"],
            "person_id": person_id,
            "vector_count": vector_count,
            "modeled_scan_speedup": scan_speedup,
            "host_service_seconds": host_service,
            "candidate_service_seconds": candidate_service,
            "host_ttft_seconds": host_ttft,
            "candidate_ttft_seconds": candidate_ttft,
            "host_retrieval_seconds": retrieval,
            "candidate_retrieval_seconds": candidate_retrieval,
            "host_materialize_seconds": materialize_host,
            "pim_materialize": materialize,
            "capacity_admitted": int(row["packed_bytes"]) <= 2 * 1024 ** 3,
        })
    host_service = [item["host_service_seconds"] for item in projected]
    candidate_service = [item["candidate_service_seconds"] for item in projected]
    host_ttft = [item["host_ttft_seconds"] for item in projected]
    candidate_ttft = [item["candidate_ttft_seconds"] for item in projected]
    p95_host = _quantile(host_service, .95)
    p95_candidate = _quantile(candidate_service, .95)
    service_speedup = sum(host_service) / sum(candidate_service)
    ttft_speedup = sum(host_ttft) / sum(candidate_ttft)
    return {
        "schema": "joint-retrieval-evidence-pim-projection-v1",
        "model_kind": "target_architecture_sensitivity_projection",
        "requests": len(projected),
        "retrieval_scan_fraction_assumed": retrieval_scan_fraction,
        "quality_contract_assumed": bool(quality_contract_assumed),
        "metrics": {
            "service_speedup": service_speedup,
            "ttft_speedup": ttft_speedup,
            "service_p95_host_seconds": p95_host,
            "service_p95_candidate_seconds": p95_candidate,
            "p95_ratio": p95_host / p95_candidate,
            "retrieval_host_seconds_total": sum(item["host_retrieval_seconds"] for item in projected),
            "retrieval_candidate_seconds_total": sum(item["candidate_retrieval_seconds"] for item in projected),
        },
        "gates": {
            "quality_contract_assumed": bool(quality_contract_assumed),
            "capacity_all_admitted": all(item["capacity_admitted"] for item in projected),
            "service_speedup_pass": service_speedup >= speedup_gate,
            "ttft_speedup_pass": ttft_speedup >= speedup_gate,
            "p95_not_worse": p95_candidate <= p95_host,
        },
        "positive_architecture_gate_pass": bool(
            quality_contract_assumed and all(item["capacity_admitted"] for item in projected)
            and service_speedup >= speedup_gate and ttft_speedup >= speedup_gate
            and p95_candidate <= p95_host),
        "rows": projected,
        "claim_boundary": (
            "server-measured request phases plus shared LPDDR architecture models; "
            "retrieval scan attribution is assumed, not observed; no device or E2E claim"),
    }
