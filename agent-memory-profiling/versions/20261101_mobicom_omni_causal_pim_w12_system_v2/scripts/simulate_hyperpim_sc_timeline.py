#!/usr/bin/env python3
"""Build an auditable cycle-level request timeline for HyperPIM-SC."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from pim_backend.hyperpim_timeline import (
    HyperPimTimelineConfig, matrix_cycles, simulate_token, with_residual_bits)


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency(values):
    if not values:
        return None
    return {
        "count": len(values),
        "sum_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
        "p99_seconds": percentile(values, 0.99),
    }


def decode_sequence(trace_root):
    paths = sorted(path for path in Path(trace_root).glob("user_*.json")
                   if not path.name.endswith(".partial.json"))
    if len(paths) != 4:
        raise ValueError("four complete measured trace files required")
    for path in paths:
        run = json.loads(path.read_text())
        if not run.get("complete") or run.get("evaluation_label_files_read") != []:
            raise ValueError("complete label-isolated trace required")
        for row in run["rows"]:
            parse = (row.get("proof_plan") or {}).get("query_parse") or {}
            tokens = int(parse.get("output_token_count") or 0)
            operators = (((row.get("proof_plan") or {}).get("retrieval") or {})
                         .get("linear_operator_trace") or {}).get("operators") or []
            if tokens <= 1 or not operators:
                continue
            steps = tokens - 1
            sequence = []
            for operator in operators:
                if operator.get("phase") != "decode" or operator.get("rows") != 1:
                    continue
                rate = operator["calls"] / steps
                if not math.isclose(rate, round(rate), abs_tol=1e-9):
                    raise ValueError("decode signature is not fixed per token")
                match = LAYER_RE.search(operator["name"])
                layer = int(match.group(1)) if match else None
                for _ in range(round(rate)):
                    sequence.append({
                        "name": operator["name"],
                        "inputs": int(operator["in_features"]),
                        "outputs": int(operator["out_features"]),
                        "layer": layer,
                    })
            if len(sequence) != 249:
                raise ValueError("expected 249 ordered decode Linear calls")
            return {
                "path": str(path),
                "question_id": row["question_id"],
                "steps": steps,
                "sequence": sequence,
            }
    raise ValueError("one ordered decode trace required")


def calibrate(sequence, matrix_costs, config):
    costs = {row["matrix_id"]: row for row in matrix_costs}
    weighted_model_ms = 0.0
    weighted_geniepim_ms = 0.0
    by_shape = {}
    counts = Counter(f"{item['outputs']}x{item['inputs']}" for item in sequence)
    for shape, count in sorted(counts.items()):
        if shape not in costs:
            raise ValueError("missing GeniePIM calibration shape: " + shape)
        outputs, inputs = (int(value) for value in shape.split("x"))
        model = matrix_cycles(inputs, outputs, config)
        geniepim_ns = float(costs[shape]["cost"]["pim_time_ns"])
        model_ms = count * model["active_ms"]
        geniepim_ms = count * geniepim_ns / 1e6
        weighted_model_ms += model_ms
        weighted_geniepim_ms += geniepim_ms
        by_shape[shape] = {
            "calls_per_token": count,
            "model_active_ms_per_token": model_ms,
            "geniepim_ms_per_token": geniepim_ms,
            "model_over_geniepim": model_ms / geniepim_ms if geniepim_ms else None,
        }
    return {
        "weighted_model_active_ms_per_token": weighted_model_ms,
        "weighted_geniepim_ms_per_token": weighted_geniepim_ms,
        "model_over_geniepim": (weighted_model_ms / weighted_geniepim_ms
                                if weighted_geniepim_ms else None),
        "by_shape": by_shape,
        "calibration_rule": "the model must stay within 0.8-1.3x weighted GeniePIM cost",
        "passed": bool(weighted_geniepim_ms
                       and 0.8 <= weighted_model_ms / weighted_geniepim_ms <= 1.3),
        "boundary": "GeniePIM is an external analytical calibration point, not a physical device",
    }

def assess_target(projection, pim_ms, nonlinear, grammar, dispatch, safety_factor):
    structured_ms = safety_factor * (pim_ms + nonlinear + grammar + dispatch)
    reader_ms = safety_factor * (pim_ms + nonlinear + dispatch)
    old_structured = float(projection["target_ms_per_token"]["structured"])
    old_reader = float(projection["target_ms_per_token"]["free_text_reader"])
    baseline_service, baseline_ttft = [], []
    control_service, control_ttft = [], []
    target_service, target_ttft = [], []
    details = []
    for row in projection["requests_detail"]:
        parse_tokens = int(row["parse_tokens"])
        grounding_tokens = int(row["grounding_tokens"])
        reader_tokens = int(row["reader_tokens"])
        old_service = float(row["target_service_seconds"])
        old_ttft = float(row["target_ttft_seconds"])
        service_delta = ((parse_tokens + grounding_tokens)
                         * (structured_ms - old_structured) / 1000)
        service_delta += reader_tokens * (reader_ms - old_reader) / 1000
        ttft_delta = ((parse_tokens + grounding_tokens)
                      * (structured_ms - old_structured) / 1000)
        service = old_service + service_delta
        ttft = old_ttft + ttft_delta
        baseline_service.append(float(row["baseline_service_seconds"]))
        baseline_ttft.append(float(row["baseline_ttft_seconds"]))
        control_service.append(float(row["control_service_seconds"]))
        control_ttft.append(float(row["control_ttft_seconds"]))
        target_service.append(service)
        target_ttft.append(ttft)
        details.append({
            **row,
            "cycle_target_service_seconds": service,
            "cycle_target_ttft_seconds": ttft,
            "service_delta_ms_from_analytical_projection": 1000 * service_delta,
            "ttft_delta_ms_from_analytical_projection": 1000 * ttft_delta,
        })
    baseline = {"service": latency(baseline_service), "ttft": latency(baseline_ttft)}
    control = {"service": latency(control_service), "ttft": latency(control_ttft)}
    target = {"service": latency(target_service), "ttft": latency(target_ttft)}
    speedup = {
        "service_vs_baseline": baseline["service"]["sum_seconds"]
        / target["service"]["sum_seconds"],
        "ttft_vs_baseline": baseline["ttft"]["sum_seconds"]
        / target["ttft"]["sum_seconds"],
        "service_p95_vs_baseline": baseline["service"]["p95_seconds"]
        / target["service"]["p95_seconds"],
        "ttft_p95_vs_baseline": baseline["ttft"]["p95_seconds"]
        / target["ttft"]["p95_seconds"],
        "service_vs_host_control_lower_bound": control["service"]["sum_seconds"]
        / target["service"]["sum_seconds"],
        "ttft_vs_host_control_lower_bound": control["ttft"]["sum_seconds"]
        / target["ttft"]["sum_seconds"],
        "service_p95_vs_host_control_lower_bound": control["service"]["p95_seconds"]
        / target["service"]["p95_seconds"],
        "ttft_p95_vs_host_control_lower_bound": control["ttft"]["p95_seconds"]
        / target["ttft"]["p95_seconds"],
    }
    return {
        "baseline": baseline,
        "control": control,
        "target": target,
        "speedup": speedup,
        "details": details,
        "structured_ms_per_token": structured_ms,
        "reader_ms_per_token": reader_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--matrix-costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safety-factor", type=float, default=1.15)
    parser.add_argument("--dispatch-ms-per-token", type=float, default=4.0)
    parser.add_argument("--cold-load-gbps", type=float, default=90.0693433912629)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh timeline output required")
    if (not math.isfinite(args.safety_factor) or args.safety_factor < 1
            or not math.isfinite(args.dispatch_ms_per_token)
            or args.dispatch_ms_per_token < 0
            or not math.isfinite(args.cold_load_gbps) or args.cold_load_gbps <= 0):
        raise ValueError("valid safety, dispatch and cold-load bandwidth required")

    projection = json.loads(args.projection.read_text())
    matrix_costs = json.loads(args.matrix_costs.read_text())
    sampled = decode_sequence(args.trace_root)
    sequence = sampled["sequence"]
    weight_elements = sum(item["inputs"] * item["outputs"] for item in sequence)
    if weight_elements != 4_204_789_760:
        raise ValueError("decoder weight inventory differs from the frozen trace")

    base = HyperPimTimelineConfig()
    calibration = calibrate(sequence, matrix_costs, base)
    if not calibration["passed"]:
        raise ValueError("cycle model failed the GeniePIM calibration gate")

    token_timelines = {}
    for bits in (0, 2, 4):
        simulated = simulate_token(sequence, with_residual_bits(base, bits))
        token_timelines[f"r{bits}"] = simulated
    selected = token_timelines["r2"]

    nonlinear = float(projection["gpu_nonlinear_profile"]["median_ms_per_token"])
    grammar = float(projection["pim_grammar_model"]["milliseconds_per_token"])
    dispatch = args.dispatch_ms_per_token
    pim_ms = selected["totals"]["pim_linear_ms_per_token"]
    structured_ms = args.safety_factor * (pim_ms + nonlinear + grammar + dispatch)
    reader_ms = args.safety_factor * (pim_ms + nonlinear + dispatch)
    old_structured = float(projection["target_ms_per_token"]["structured"])
    old_reader = float(projection["target_ms_per_token"]["free_text_reader"])

    assessment = assess_target(projection, pim_ms, nonlinear, grammar, dispatch,
                               args.safety_factor)
    baseline = assessment["baseline"]
    control = assessment["control"]
    target = assessment["target"]
    details = assessment["details"]

    variant_configs = {
        "base": with_residual_bits(base, 2),
        "residual_mac_2x": replace(
            with_residual_bits(base, 2), residual_mac_relative_cost=2.0),
        "conversion_lanes_8": replace(
            with_residual_bits(base, 2), conversion_lanes_per_bank=8),
        "refresh_block_2x": replace(
            with_residual_bits(base, 2),
            refresh_block_cycles=2 * base.refresh_block_cycles),
        "memory_mac_overlap": replace(
            with_residual_bits(base, 2), serialize_bank_memory_and_mac=False),
        "joint_worst": replace(
            with_residual_bits(base, 2), residual_mac_relative_cost=2.0,
            conversion_lanes_per_bank=8,
            refresh_block_cycles=2 * base.refresh_block_cycles,
            t_rcd_cycles=2 * base.t_rcd_cycles,
            t_rp_cycles=2 * base.t_rp_cycles,
            layer_fence_cycles=2 * base.layer_fence_cycles),
    }
    variant_dispatch = {name: dispatch for name in variant_configs}
    variant_dispatch["joint_worst"] = 10.0
    sensitivity = {}
    for name, config in variant_configs.items():
        simulated = simulate_token(sequence, config)
        variant = assess_target(
            projection, simulated["totals"]["pim_linear_ms_per_token"],
            nonlinear, grammar, variant_dispatch[name], args.safety_factor)
        sensitivity[name] = {
            "config": simulated["config"],
            "dispatch_ms_per_token": variant_dispatch[name],
            "pim_linear_ms_per_token": simulated["totals"]["pim_linear_ms_per_token"],
            "service_speedup_vs_host_control_lower_bound": variant["speedup"][
                "service_vs_host_control_lower_bound"],
            "ttft_speedup_vs_host_control_lower_bound": variant["speedup"][
                "ttft_vs_host_control_lower_bound"],
            "service_p95_speedup_vs_host_control_lower_bound": variant["speedup"][
                "service_p95_vs_host_control_lower_bound"],
            "ttft_p95_speedup_vs_host_control_lower_bound": variant["speedup"][
                "ttft_p95_vs_host_control_lower_bound"],
        }

    resident_bytes = int(projection["decode_token_accounting"]["pim_resident_w8_weight_bytes"])
    resident_bytes += int(projection["decode_token_accounting"]["pim_resident_residual_bytes"])
    cold_load_seconds = resident_bytes / (args.cold_load_gbps * 1e9)
    result = {
        "schema": "hyperpim-sc-cycle-request-timeline-v1",
        "requests": len(details),
        "boundary": (
            "target LPDDR bank-PIM cycle timeline calibrated to GeniePIM matrix costs; "
            "service/TTFT are anchored to measured server stages but are not a physical "
            "PIM or phone measurement"),
        "inputs_sha256": {
            str(args.projection): digest(args.projection),
            str(args.matrix_costs): digest(args.matrix_costs),
            sampled["path"]: digest(sampled["path"]),
        },
        "decode_sequence": {
            "source": sampled["path"],
            "question_id": sampled["question_id"],
            "steps": sampled["steps"],
            "calls_per_token": len(sequence),
            "weight_elements": weight_elements,
            "layers": len({item["layer"] for item in sequence if item["layer"] is not None}),
        },
        "calibration": calibration,
        "token_timelines": token_timelines,
        "selected_arm": {
            "name": "w8a16_r2",
            "pim_linear_ms_per_token": pim_ms,
            "structured_ms_per_token": structured_ms,
            "reader_ms_per_token": reader_ms,
            "analytical_projection_structured_ms_per_token": old_structured,
            "analytical_projection_reader_ms_per_token": old_reader,
            "cycle_delta_ms_per_token": pim_ms - float(
                projection["projected_token_cost_ms"]["pim_linear"]),
        },
        "stage_token_accounting": {
            "parser_tokens": int(projection["decode_token_accounting"]["parser"]),
            "grounding_tokens": int(projection["decode_token_accounting"]["grounding"]),
            "reader_tokens": int(projection["decode_token_accounting"]["reader"]),
            "total_tokens": int(projection["decode_token_accounting"]["total"]),
            "pim_linear_seconds": float(
                projection["decode_token_accounting"]["total"]) * pim_ms / 1000,
            "gpu_nonlinear_seconds": float(
                projection["decode_token_accounting"]["total"]) * nonlinear / 1000,
            "grammar_scan_seconds": (
                float(projection["decode_token_accounting"]["parser"])
                + float(projection["decode_token_accounting"]["grounding"]))
            * grammar / 1000,
            "dispatch_seconds": float(
                projection["decode_token_accounting"]["total"]) * dispatch / 1000,
        },
        "cold_start": {
            "resident_bytes": resident_bytes,
            "load_bandwidth_gbps": args.cold_load_gbps,
            "load_seconds": cold_load_seconds,
            "policy": "amortized once per resident model image; not repeated per request",
        },
        "scenarios": {
            "baseline_measured": baseline,
            "host_control_lower_bound": control,
            "analytical_projection": projection["scenarios"]["hyperpim_sc_target"],
            "cycle_target": target,
            "cycle_target_speedup": assessment["speedup"],
        },
        "cycle_sensitivity": sensitivity,
        "requests_detail": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "requests": result["requests"],
        "calibration": {
            "model_over_geniepim": calibration["model_over_geniepim"],
            "passed": calibration["passed"],
        },
        "token_ms": {
            "r0": token_timelines["r0"]["totals"]["pim_linear_ms_per_token"],
            "r2": token_timelines["r2"]["totals"]["pim_linear_ms_per_token"],
            "r4": token_timelines["r4"]["totals"]["pim_linear_ms_per_token"],
        },
        "selected": result["selected_arm"],
        "speedup_vs_host_control_lower_bound": {
            key: result["scenarios"]["cycle_target_speedup"][key]
            for key in ("service_vs_host_control_lower_bound",
                        "ttft_vs_host_control_lower_bound",
                        "service_p95_vs_host_control_lower_bound",
                        "ttft_p95_vs_host_control_lower_bound")
        },
        "cycle_sensitivity": {
            name: {
                "pim_linear_ms_per_token": value["pim_linear_ms_per_token"],
                "service_vs_host_control_lower_bound":
                    value["service_speedup_vs_host_control_lower_bound"],
                "ttft_vs_host_control_lower_bound":
                    value["ttft_speedup_vs_host_control_lower_bound"],
            }
            for name, value in sensitivity.items()
        },
        "cold_load_seconds": cold_load_seconds,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
