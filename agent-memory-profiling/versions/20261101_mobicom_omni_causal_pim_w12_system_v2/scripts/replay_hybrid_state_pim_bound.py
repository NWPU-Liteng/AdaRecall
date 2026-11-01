"""Replay the hybrid-state PIM route without hiding full-attention KV cost.

The earlier workload replay removed the complete Host restore/eviction phase.
That is not consistent with the active design: PIM owns only the linear-
attention recurrent/conv state, while full-attention KV remains Host/GPU-owned.
This script keeps the full-attention share of every measured transfer and
reports a conservative capacity-only result separately from a phase-model
substitution bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("nonempty values required")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    baseline_total = sum(baseline)
    candidate_total = sum(candidate)
    baseline_p95 = percentile(baseline, 0.95)
    candidate_p95 = percentile(candidate, 0.95)
    return {
        "accesses": len(baseline),
        "baseline_total_seconds": baseline_total,
        "candidate_total_seconds": candidate_total,
        "baseline_mean_seconds": statistics.fmean(baseline),
        "candidate_mean_seconds": statistics.fmean(candidate),
        "baseline_p95_seconds": baseline_p95,
        "candidate_p95_seconds": candidate_p95,
        "speedup_total": baseline_total / candidate_total if candidate_total else None,
        "speedup_mean": statistics.fmean(baseline) / statistics.fmean(candidate),
        "speedup_p95": baseline_p95 / candidate_p95 if candidate_p95 else None,
    }


def replay(
    churn_path: Path,
    qualification_path: Path,
    decode_components_path: Path,
    output: Path,
) -> dict[str, Any]:
    churn = json.loads(churn_path.read_text())
    qualification = json.loads(qualification_path.read_text())
    decode_components = json.loads(decode_components_path.read_text())
    if churn.get("status") != "complete" or not churn.get("access_records"):
        raise ValueError("complete state-churn access records required")

    native = qualification.get("native_state", {})
    total_state_bytes = int(native.get("total_bytes", 0))
    linear_state_bytes = int(native.get("linear_attention_state_bytes", 0))
    if total_state_bytes <= 0 or linear_state_bytes <= 0:
        raise ValueError("qualification must expose positive native-state byte counts")
    if linear_state_bytes > total_state_bytes:
        raise ValueError("linear state cannot exceed native state")

    linear_fraction = linear_state_bytes / total_state_bytes
    full_attention_fraction = 1.0 - linear_fraction
    fraction = decode_components.get("derived", {}).get("extended_pim_fraction_upper_mean")
    if fraction is None or not 0 <= float(fraction) < 1:
        raise ValueError("finite hybrid-state upper fraction required")
    plan = qualification.get("pim_plan", {})
    pim_step = plan.get("pim_step", {})
    clock_hz = float(qualification.get("pim_profile", {}).get("clock_hz", 0.0))
    if not clock_hz:
        clock_hz = 800_000_000.0
    pim_step_seconds = float(pim_step.get("cycles", 0.0)) / clock_hz

    rows: list[dict[str, Any]] = []
    baseline: list[float] = []
    capacity_only: list[float] = []
    phase_substitution: list[float] = []
    for access in churn["access_records"]:
        baseline_total = float(access["total_seconds"])
        baseline_decode = float(access["decode_seconds"])
        restore = float(access["restore_seconds"])
        eviction = float(access["eviction_seconds"])

        # Only the linear recurrent/conv portion avoids Host transfer.  The
        # full-attention KV share remains a real miss cost in this design.
        retained_restore = restore * full_attention_fraction
        retained_eviction = eviction * full_attention_fraction
        capacity_total = baseline_decode + retained_restore + retained_eviction

        # This second column is deliberately a model bound, not a measurement:
        # it replaces the profiled eligible decode fraction with the current
        # uncalibrated PIM phase model while retaining full-KV transfers.
        modeled_decode = baseline_decode * (1.0 - float(fraction)) + pim_step_seconds
        phase_total = modeled_decode + retained_restore + retained_eviction

        baseline.append(baseline_total)
        capacity_only.append(capacity_total)
        phase_substitution.append(phase_total)
        rows.append({
            "access_index": access["access_index"],
            "agent_index": access["agent_index"],
            "cycle_index": access["cycle_index"],
            "cache_hit": access["cache_hit"],
            "baseline_total_seconds": baseline_total,
            "baseline_decode_seconds": baseline_decode,
            "baseline_restore_seconds": restore,
            "baseline_eviction_seconds": eviction,
            "retained_full_attention_restore_seconds": retained_restore,
            "retained_full_attention_eviction_seconds": retained_eviction,
            "capacity_only_total_seconds": capacity_total,
            "phase_model_decode_seconds": modeled_decode,
            "phase_substitution_total_seconds": phase_total,
            "removed_linear_state_transfer_seconds": (
                restore + eviction
            ) * linear_fraction,
        })

    result = {
        "schema": "hybrid-state-pim-workload-bound-v1",
        "inputs": {
            "churn_sha256": digest(churn_path),
            "qualification_sha256": digest(qualification_path),
            "decode_components_sha256": digest(decode_components_path),
        },
        "workload": {
            "schedule": churn.get("schedule"),
            "agent_count": churn.get("agent_count"),
            "target_tokens": churn.get("target_tokens"),
            "hbm_capacity_agents": churn.get("hbm_capacity_agents"),
            "accesses": len(rows),
        },
        "state_scope": {
            "total_native_state_bytes": total_state_bytes,
            "linear_attention_state_bytes": linear_state_bytes,
            "linear_state_fraction": linear_fraction,
            "full_attention_fraction_retained_on_host": full_attention_fraction,
        },
        "model": {
            "hybrid_state_upper_fraction": float(fraction),
            "pim_step_cycles": pim_step.get("cycles"),
            "pim_step_seconds": pim_step_seconds,
            "physical_pim_executed": False,
            "measured_candidate_run": False,
        },
        "summary": {
            "baseline": summarize(baseline, baseline),
            "capacity_only_linear_state": summarize(baseline, capacity_only),
            "phase_substitution_bound": summarize(baseline, phase_substitution),
        },
        "rows": rows,
        "boundary": (
            "measured Host state-churn trace replay. The capacity-only column removes "
            "only the linear recurrent/conv share of measured restore/eviction and "
            "therefore retains full-attention KV cost. The phase-substitution column "
            "also replaces an upper-bound decode fraction with an uncalibrated PIM "
            "phase model. Neither column is physical PIM or a measured E2E pair."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--churn", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--decode-components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    result = replay(args.churn, args.qualification, args.decode_components, args.output)
    print(json.dumps({"output": str(args.output), **result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
