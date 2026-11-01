"""Replay a capacity-aware PIM route for the pre-reader output projection only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pim_backend.decode_gemv import gemv_cost
from pim_backend.lpddr_bank_pim import BankPimConfig


TARGET = "lm_head"


def _trace_operator(trace, name):
    matches = [item for item in trace.get("operators", [])
               if item.get("phase") == "decode" and item.get("rows") == 1
               and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one decode trace for {name}, found {len(matches)}")
    item = matches[0]
    expected_bytes = item["in_features"] * item["out_features"] * item["element_bytes"]
    if item.get("weight_bytes") != expected_bytes or item.get("calls", 0) <= 0:
        raise ValueError(f"invalid decode trace for {name}")
    return item


def project_structured_call(trace, config, *, target=TARGET):
    operators = [item for item in trace.get("operators", [])
                 if item.get("phase") == "decode" and item.get("rows") == 1]
    if not operators:
        raise ValueError("structured call has no batch-1 decode Linear trace")
    target_operator = _trace_operator(trace, target)
    target_cost = gemv_cost(target_operator["in_features"], target_operator["out_features"],
                            target_operator["element_bytes"], config)
    all_host_cycles = 0
    for item in operators:
        expected_bytes = item["in_features"] * item["out_features"] * item["element_bytes"]
        if item.get("weight_bytes") != expected_bytes or item.get("calls", 0) <= 0:
            raise ValueError("invalid decode Linear trace")
        cost = gemv_cost(item["in_features"], item["out_features"],
                         item["element_bytes"], config)
        all_host_cycles += item["calls"] * cost["host_cycles"]
    target_host_cycles = target_operator["calls"] * target_cost["host_cycles"]
    target_pim_cycles = target_operator["calls"] * target_cost["pim_cycles"]
    admitted = target_cost["capacity_feasible"] and target_cost["buffer_feasible"]
    warm_hybrid_cycles = (all_host_cycles - target_host_cycles + target_pim_cycles
                          if admitted else all_host_cycles)
    route = "pim" if admitted else "host_capacity_fallback"
    return {
        "target": target,
        "calls": target_operator["calls"],
        "inputs": target_operator["in_features"],
        "outputs": target_operator["out_features"],
        "element_bytes": target_operator["element_bytes"],
        "resident_bytes": target_cost["resident_bytes"],
        "capacity_bytes": config.pim_capacity_bytes,
        "capacity_feasible": target_cost["capacity_feasible"],
        "buffer_feasible": target_cost["buffer_feasible"],
        "all_linear_host_cycles": all_host_cycles,
        "target_host_cycles": target_host_cycles,
        "target_pim_cycles": target_pim_cycles,
        "warm_hybrid_cycles": warm_hybrid_cycles,
        "one_time_placement_cycles": target_cost["stage_cycles"] if admitted else 0,
        "cold_hybrid_cycles": (warm_hybrid_cycles + target_cost["stage_cycles"]
                               if admitted else all_host_cycles),
        "target_host_external_bytes": target_operator["calls"] * target_cost["host_external_bytes"],
        "target_pim_external_bytes": target_operator["calls"] * target_cost["pim_external_bytes"],
        "target_pim_weight_internal_bytes": target_operator["calls"] * target_cost["pim_weight_internal_bytes"],
        "linear_phase_speedup_warm": all_host_cycles / warm_hybrid_cycles,
        "linear_phase_speedup_cold": all_host_cycles / (warm_hybrid_cycles + (
            target_cost["stage_cycles"] if admitted else 0)),
        "target_weight_host_fraction_of_linear_model": target_host_cycles / all_host_cycles,
        "route": route,
        "boundary": "modeled output-head linear phase only; not an end-to-end latency result",
    }


def _structured_calls(run):
    for row in run["rows"]:
        retrieval = row["proof_plan"]["retrieval"]
        yield row, "query_parser", retrieval.get("linear_operator_trace"), {
            "profile_wall_seconds": retrieval["timing"].get("query_parser_total_seconds"),
            "profile_wall_comparable": False,
        }
        for index, attempt in enumerate(row["proof_plan"]["grounding"].get("frontier_attempts", [])):
            yield row, f"source_grounding_{index}", attempt.get("linear_operator_trace"), {
                "profile_wall_seconds": attempt.get("reader_timing", {}).get("prefill_seconds", 0.0)
                + attempt.get("reader_timing", {}).get("decode_seconds", 0.0),
                "profile_wall_comparable": False,
            }


def project_run(run, profile, *, pim_capacity_bytes=None):
    if not run.get("complete") or not run.get("pre_reader_operator_profile", {}).get("enabled"):
        raise ValueError("a complete pre-reader operator-profile run is required")
    if run.get("evaluation_label_files_read") != []:
        raise ValueError("profiled run must not read evaluation labels")
    profile_config = dict(profile["config"])
    configured_capacity_bytes = int(profile_config["pim_capacity_bytes"])
    if pim_capacity_bytes is not None:
        if type(pim_capacity_bytes) is not int or pim_capacity_bytes <= 0:
            raise ValueError("PIM capacity override must be a positive integer byte count")
        profile_config["pim_capacity_bytes"] = pim_capacity_bytes
    config = BankPimConfig(**profile_config)
    calls = []
    for row, phase, trace, timing in _structured_calls(run):
        if trace is None:
            continue
        projection = project_structured_call(trace, config)
        projection.update({"person_id": row["person_id"], "question_id": row["question_id"],
                           "phase": phase, **timing})
        calls.append(projection)
    if not calls:
        raise ValueError("no eligible structured calls found")
    target = calls[0]
    if any((item["inputs"], item["outputs"], item["element_bytes"], item["resident_bytes"])
           != (target["inputs"], target["outputs"], target["element_bytes"], target["resident_bytes"])
           for item in calls):
        raise ValueError("lm_head shape changes across structured calls")
    all_host = sum(item["all_linear_host_cycles"] for item in calls)
    warm = sum(item["warm_hybrid_cycles"] for item in calls)
    cold = warm + target["one_time_placement_cycles"]
    target_host = sum(item["target_host_cycles"] for item in calls)
    target_pim = sum(item["target_pim_cycles"] for item in calls)
    return {
        "schema_version": "pre-reader-lm-head-pim-replay-v1",
        "profile_name": profile["name"],
        "profile_calibration_status": profile["calibration_status"],
        "structured_call_count": len(calls),
        "capacity": {
            "target": TARGET,
            "resident_bytes": target["resident_bytes"],
            "declared_pim_pool_bytes": config.pim_capacity_bytes,
            "profile_declared_pim_pool_bytes": configured_capacity_bytes,
            "capacity_override_bytes": pim_capacity_bytes,
            "capacity_override_applied": pim_capacity_bytes is not None,
            "pool_utilization": target["resident_bytes"] / config.pim_capacity_bytes,
            "capacity_feasible": all(item["capacity_feasible"] for item in calls),
            "buffer_feasible": all(item["buffer_feasible"] for item in calls),
            "route": "pim" if all(item["route"] == "pim" for item in calls)
            else "host_capacity_fallback",
        },
        "aggregate": {
            "all_linear_host_cycles": all_host,
            "target_host_cycles": target_host,
            "target_pim_cycles": target_pim,
            "warm_hybrid_cycles": warm,
            "cold_hybrid_cycles": cold,
            "one_time_placement_cycles": target["one_time_placement_cycles"],
            "linear_phase_speedup_warm": all_host / warm,
            "linear_phase_speedup_cold": all_host / cold,
            "target_weight_host_fraction_of_linear_model": target_host / all_host,
            "target_host_external_bytes": sum(item["target_host_external_bytes"] for item in calls),
            "target_pim_external_bytes": sum(item["target_pim_external_bytes"] for item in calls),
        },
        "calls": calls,
        "end_to_end_speedup": None,
        "decision_boundary": {
            "physical_pim_execution": False,
            "profile_wall_times_performance_comparable": False,
            "quality_contract_rechecked": False,
            "requires_next_step": [
                "functional lm_head dispatch with exact logits contract",
                "cold/warm placement and shared-bus contention accounting",
                "paired no-profile E2E measurement on the same questions",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiled-run", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    capacity_group = parser.add_mutually_exclusive_group()
    capacity_group.add_argument("--pim-capacity-bytes", type=int)
    capacity_group.add_argument("--pim-capacity-gib", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh replay output required")
    run = json.loads(args.profiled_run.read_text())
    profile = json.loads(args.profile.read_text())
    if args.pim_capacity_gib is not None:
        if args.pim_capacity_gib <= 0:
            raise ValueError("PIM capacity in GiB must be positive")
        capacity_override = int(args.pim_capacity_gib * 1024**3)
    else:
        capacity_override = args.pim_capacity_bytes
    result = project_run(run, profile, pim_capacity_bytes=capacity_override)
    result["profiled_run_sha256"] = hashlib.sha256(args.profiled_run.read_bytes()).hexdigest()
    result["profile_sha256"] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result["aggregate"].get(key) for key in (
        "linear_phase_speedup_warm", "linear_phase_speedup_cold",
        "target_weight_host_fraction_of_linear_model")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
