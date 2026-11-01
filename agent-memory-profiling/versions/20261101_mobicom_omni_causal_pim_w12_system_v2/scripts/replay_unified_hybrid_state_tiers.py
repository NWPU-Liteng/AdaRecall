"""Replay a GPU/PIM/Host state hierarchy over frozen state-churn accesses.

The replay reports residency and miss counts only. PIM service timing is left
unreported because Qwen3.5's exact state-kernel integration and device
calibration are not yet available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replay(churn_path: Path, qualification_path: Path, output: Path) -> dict[str, Any]:
    churn = json.loads(churn_path.read_text())
    qualification = json.loads(qualification_path.read_text())
    records = churn.get("access_records")
    if churn.get("status") != "complete" or not isinstance(records, list) or not records:
        raise ValueError("complete state-churn accesses required")

    plan = qualification.get("pim_plan", {})
    admission = plan.get("admission", {})
    if admission.get("state_scope") != "unified":
        raise ValueError("unified complete-state PIM capacity qualification required")
    if not admission.get("available_bytes") or not admission.get("per_agent_state_bytes"):
        raise ValueError("unified PIM capacity fields missing")

    gpu_capacity = int(churn["hbm_capacity_agents"])
    pim_capacity = int(admission["resident_agents"])
    if gpu_capacity <= 0 or pim_capacity < 0:
        raise ValueError("invalid GPU/PIM state capacities")

    pim_profile = qualification.get("pim_profile", {})
    pim_clock_hz = float(pim_profile.get("clock_hz", 0.0))
    pim_external_bytes_per_cycle = float(
        pim_profile.get("external_service_bytes_per_cycle", 0.0)
    )
    if pim_clock_hz <= 0 or pim_external_bytes_per_cycle <= 0:
        raise ValueError("PIM external transfer profile missing from qualification")

    gpu: OrderedDict[int, None] = OrderedDict()
    pim: OrderedDict[int, None] = OrderedDict()
    rows: list[dict[str, Any]] = []
    counts = {
        "gpu_hits": 0,
        "pim_served": 0,
        "host_restores": 0,
        "gpu_to_pim_promotions": 0,
        "pim_to_host_writebacks": 0,
    }
    state_bytes = int(admission["per_agent_state_bytes"])
    promoted_bytes = 0
    writeback_bytes = 0

    for access in records:
        agent = int(access["agent_index"])
        source: str
        evicted_gpu = None
        evicted_pim = None
        if agent in gpu:
            source = "gpu"
            counts["gpu_hits"] += 1
            gpu.move_to_end(agent)
        elif agent in pim:
            source = "pim"
            counts["pim_served"] += 1
            pim.move_to_end(agent)
        else:
            source = "host"
            counts["host_restores"] += 1
            if len(gpu) >= gpu_capacity:
                evicted_gpu, _ = gpu.popitem(last=False)
                if pim_capacity:
                    if evicted_gpu in pim:
                        pim.move_to_end(evicted_gpu)
                    else:
                        if len(pim) >= pim_capacity:
                            evicted_pim, _ = pim.popitem(last=False)
                            counts["pim_to_host_writebacks"] += 1
                            writeback_bytes += state_bytes
                        counts["gpu_to_pim_promotions"] += 1
                        promoted_bytes += state_bytes
                        pim[evicted_gpu] = None
            gpu[agent] = None

        rows.append({
            "access_index": int(access["access_index"]),
            "agent_index": agent,
            "source_tier": source,
            "gpu_evicted_agent": evicted_gpu,
            "pim_evicted_to_host_agent": evicted_pim,
            "baseline_host_restore_seconds": float(access["restore_seconds"]),
            "baseline_host_eviction_seconds": float(access["eviction_seconds"]),
        })

    observed_baseline_hits = sum(bool(item["cache_hit"]) for item in records)
    if counts["gpu_hits"] != observed_baseline_hits:
        raise ValueError(
            "tier replay GPU-hit count diverged from measured baseline; "
            "baseline cache policy or initial state assumptions differ"
        )
    baseline_host_misses = len(records) - observed_baseline_hits
    host_miss_reduction = baseline_host_misses - counts["host_restores"]
    migration_bytes = promoted_bytes + writeback_bytes
    modeled_migration_seconds = migration_bytes / (
        pim_external_bytes_per_cycle * pim_clock_hz
    )
    result = {
        "schema": "unified-hybrid-state-tier-replay-v1",
        "inputs": {
            "churn_sha256": digest(churn_path),
            "qualification_sha256": digest(qualification_path),
        },
        "workload": {
            "schedule": churn.get("schedule"),
            "agent_count": churn.get("agent_count"),
            "target_tokens": churn.get("target_tokens"),
            "accesses": len(records),
            "gpu_capacity_states": gpu_capacity,
            "pim_capacity_complete_states": pim_capacity,
        },
        "state_bytes_per_agent": state_bytes,
        "pim_transfer_model": {
            "external_service_bytes_per_cycle": pim_external_bytes_per_cycle,
            "clock_hz": pim_clock_hz,
            "service_gb_s": pim_external_bytes_per_cycle * pim_clock_hz / 1e9,
            "calibration_status": pim_profile.get("calibration_status"),
            "boundary": "uncalibrated raw transfer lower bound; excludes command, contention, fencing and decode",
        },
        "baseline": {
            "gpu_hits": observed_baseline_hits,
            "host_misses": baseline_host_misses,
            "host_restore_seconds": sum(float(x["restore_seconds"]) for x in records),
            "host_eviction_seconds": sum(float(x["eviction_seconds"]) for x in records),
            "service_seconds": sum(float(x["total_seconds"]) for x in records),
        },
        "tiered_candidate_counts": {
            **counts,
            "host_misses_avoided": host_miss_reduction,
            "host_miss_reduction_fraction": (
                host_miss_reduction / baseline_host_misses if baseline_host_misses else 0.0
            ),
            "host_transfer_bytes_avoided_upper_bound": max(0, host_miss_reduction) * 2 * state_bytes,
            "gpu_to_pim_migration_bytes": promoted_bytes,
            "pim_to_host_writeback_bytes": writeback_bytes,
            "total_pim_tier_migration_bytes": migration_bytes,
            "raw_migration_transfer_lower_bound_seconds": modeled_migration_seconds,
            "candidate_service_seconds": None,
            "candidate_speedup": None,
        },
        "rows": rows,
        "claim_boundary": (
            "frozen trace residency replay only. Full native state consumes PIM capacity; "
            "PIM-served decode timing, GPU/PIM contention, link traffic and end-to-end "
            "service latency are not measured or inferred."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    result = replay(args.churn, args.qualification, args.output)
    print(json.dumps({
        "output": str(args.output),
        "workload": result["workload"],
        "baseline": result["baseline"],
        "tiered_candidate_counts": result["tiered_candidate_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
