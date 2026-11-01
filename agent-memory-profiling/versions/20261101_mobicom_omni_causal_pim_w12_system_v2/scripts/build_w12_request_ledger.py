#!/usr/bin/env python3
"""Build a request-level W12 bank-PIM ledger on the measured 200-request trace."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import simulate_hyperpim_sc_timeline as timeline
from pim_backend.hyperpim_timeline import HyperPimTimelineConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latency(values):
    return timeline.latency(values)


def stage_rows(projection, timeline_document, arm_name):
    arm = timeline_document["arms"][arm_name]
    nonlinear = float(projection["gpu_nonlinear_profile"]["median_ms_per_token"])
    grammar = float(projection["pim_grammar_model"]["milliseconds_per_token"])
    denominator = (arm["pim_linear_ms_per_token"] + nonlinear + grammar
                   + arm["dispatch_ms_per_token"])
    safety = arm["structured_ms_per_token"] / denominator
    if not math.isclose(safety, 1.15, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("unexpected safety factor in the frozen W12 projection")
    assessment = timeline.assess_target(
        projection, arm["pim_linear_ms_per_token"], nonlinear, grammar,
        arm["dispatch_ms_per_token"], safety)
    return arm, assessment, safety


def measured_rows(raw_root):
    result = {}
    for path in sorted(Path(raw_root).glob("user_*.json")):
        if path.name.endswith(".partial.json"):
            continue
        document = json.loads(path.read_text())
        for row in document.get("rows", []):
            result[(str(row["person_id"]), row["question_id"])] = {
                "prefill_seconds": float(row["prefill_seconds"]),
                "decode_seconds": float(row["decode_seconds"]),
                "ttft_seconds": float(row["ttft_seconds"]),
                "service_seconds": float(row["service_seconds"]),
                "stop_reason": row.get("stop_reason"),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-gain-floor", type=float, default=1.05)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh request-ledger output required")
    if not math.isfinite(args.service_gain_floor) or args.service_gain_floor <= 1:
        raise ValueError("service-gain floor must be finite and greater than one")

    projection = json.loads(args.projection.read_text())
    timeline_document = json.loads(args.timeline.read_text())
    target_arm, target_assessment, target_safety = stage_rows(
        projection, timeline_document, "w12a16")
    worst_arm, worst_assessment, worst_safety = stage_rows(
        projection, timeline_document, "w12a16_joint_worst")
    if not math.isclose(target_safety, worst_safety, rel_tol=0, abs_tol=1e-12):
        raise ValueError("target and joint-worst safety factors differ")

    inventory = projection["decode_token_accounting"]["decoder_bank_residency"]["matrices"]
    matrix_count = len(inventory)
    rows_per_token = sum(int(row["out_features"]) for row in inventory)
    host_bf16_weight_bytes_per_token = sum(
        int(row["out_features"]) * int(row["in_features"]) * 2
        for row in inventory)
    w12_logical_weight_bytes_per_token = sum(
        int(row["out_features"])
        * math.ceil(int(row["in_features"]) * 12 / 8)
        for row in inventory)
    totals = target_arm["totals"]
    config = HyperPimTimelineConfig()
    refresh_ms_per_token = (1000 * totals["refresh_cycles"] / config.clock_hz)
    fence_ms_per_token = (1000 * totals["layer_fence_cycles"] / config.clock_hz)
    dispatch_ms_per_token = float(target_arm["dispatch_ms_per_token"])
    external_bytes_per_token = int(totals["external_transfer_bytes"])
    internal_bank_bytes_per_token = int(totals["weight_bytes"])
    measured = measured_rows(args.raw_root) if args.raw_root is not None else {}

    baseline_service, baseline_ttft = [], []
    control_service, control_ttft = [], []
    target_service, target_ttft = [], []
    worst_service, worst_ttft = [], []
    stages = Counter()
    requests = []
    for target, worst in zip(target_assessment["details"],
                             worst_assessment["details"]):
        if (target["question_id"] != worst["question_id"]
                or target["person_id"] != worst["person_id"]):
            raise ValueError("W12 target and joint-worst request order differs")
        parse_tokens = int(target["parse_tokens"])
        grounding_tokens = int(target["grounding_tokens"])
        reader_tokens = int(target["reader_tokens"])
        decode_tokens = parse_tokens + grounding_tokens + reader_tokens
        if decode_tokens <= 0:
            raise ValueError("request without decoder tokens")
        stages.update({"parse_tokens": parse_tokens,
                       "grounding_tokens": grounding_tokens,
                       "reader_tokens": reader_tokens})
        key = (str(target["person_id"]), target["question_id"])
        measured_row = measured.get(key)
        if measured is not None and measured and measured_row is None:
            raise ValueError("measured request missing from the raw trace")

        stage_events = []
        for name, tokens, cost_ms in (
                ("parser", parse_tokens, target_arm["structured_ms_per_token"]),
                ("grounding", grounding_tokens, target_arm["structured_ms_per_token"]),
                ("reader", reader_tokens, target_arm["reader_ms_per_token"])):
            if tokens <= 0:
                continue
            stage_events.append({
                "stage": name,
                "decode_tokens": tokens,
                "linear_calls": tokens * matrix_count,
                "weight_rows_read": tokens * rows_per_token,
                "internal_bank_weight_bytes": tokens * internal_bank_bytes_per_token,
                "external_activation_output_bytes": tokens * external_bytes_per_token,
                "refresh_cycles": tokens * int(totals["refresh_cycles"]),
                "layer_fence_cycles": tokens * int(totals["layer_fence_cycles"]),
                "refresh_ms": tokens * refresh_ms_per_token,
                "fence_ms": tokens * fence_ms_per_token,
                "dispatch_ms": tokens * dispatch_ms_per_token,
                "projected_stage_seconds": tokens * cost_ms / 1000,
            })

        baseline_service.append(float(target["baseline_service_seconds"]))
        baseline_ttft.append(float(target["baseline_ttft_seconds"]))
        control_service.append(float(target["control_service_seconds"]))
        control_ttft.append(float(target["control_ttft_seconds"]))
        target_service.append(float(target["cycle_target_service_seconds"]))
        target_ttft.append(float(target["cycle_target_ttft_seconds"]))
        worst_service.append(float(worst["cycle_target_service_seconds"]))
        worst_ttft.append(float(worst["cycle_target_ttft_seconds"]))

        host_stream = decode_tokens * host_bf16_weight_bytes_per_token
        w12_stream = decode_tokens * w12_logical_weight_bytes_per_token
        pim_external = decode_tokens * external_bytes_per_token
        requests.append({
            "person_id": str(target["person_id"]),
            "question_id": target["question_id"],
            "decode_tokens": {
                "parser": parse_tokens,
                "grounding": grounding_tokens,
                "reader": reader_tokens,
                "total": decode_tokens,
            },
            "stage_events": stage_events,
            "external_bytes": {
                "host_bf16_weight_stream": host_stream,
                "w12_logical_weight_stream": w12_stream,
                "pim_activation_and_output": pim_external,
                "reduction_vs_host_bf16": host_stream / max(1, pim_external),
                "reduction_vs_w12_logical": w12_stream / max(1, pim_external),
            },
            "timing": {
                "baseline_service_seconds": baseline_service[-1],
                "baseline_ttft_seconds": baseline_ttft[-1],
                "host_control_service_seconds": control_service[-1],
                "host_control_ttft_seconds": control_ttft[-1],
                "pim_target_service_seconds": target_service[-1],
                "pim_target_ttft_seconds": target_ttft[-1],
                "pim_joint_worst_service_seconds": worst_service[-1],
                "pim_joint_worst_ttft_seconds": worst_ttft[-1],
                "measured_prefill_seconds": (measured_row["prefill_seconds"]
                                             if measured_row else None),
                "measured_decode_seconds": (measured_row["decode_seconds"]
                                            if measured_row else None),
            },
        })

    columns = {
        "host_only_measured": {
            "service": latency(baseline_service),
            "ttft": latency(baseline_ttft),
        },
        "host_control_lower_bound": {
            "service": latency(control_service),
            "ttft": latency(control_ttft),
        },
        "pim_target": {
            "service": latency(target_service),
            "ttft": latency(target_ttft),
        },
        "pim_joint_worst": {
            "service": latency(worst_service),
            "ttft": latency(worst_ttft),
        },
    }
    speedup = {
        "target_service_vs_host_control": (
            columns["host_control_lower_bound"]["service"]["sum_seconds"]
            / columns["pim_target"]["service"]["sum_seconds"]),
        "target_ttft_vs_host_control": (
            columns["host_control_lower_bound"]["ttft"]["sum_seconds"]
            / columns["pim_target"]["ttft"]["sum_seconds"]),
        "target_service_p95_vs_host_control": (
            columns["host_control_lower_bound"]["service"]["p95_seconds"]
            / columns["pim_target"]["service"]["p95_seconds"]),
        "target_ttft_p95_vs_host_control": (
            columns["host_control_lower_bound"]["ttft"]["p95_seconds"]
            / columns["pim_target"]["ttft"]["p95_seconds"]),
        "joint_worst_service_vs_host_control": (
            columns["host_control_lower_bound"]["service"]["sum_seconds"]
            / columns["pim_joint_worst"]["service"]["sum_seconds"]),
        "joint_worst_ttft_vs_host_control": (
            columns["host_control_lower_bound"]["ttft"]["sum_seconds"]
            / columns["pim_joint_worst"]["ttft"]["sum_seconds"]),
        "joint_worst_service_p95_vs_host_control": (
            columns["host_control_lower_bound"]["service"]["p95_seconds"]
            / columns["pim_joint_worst"]["service"]["p95_seconds"]),
        "joint_worst_ttft_p95_vs_host_control": (
            columns["host_control_lower_bound"]["ttft"]["p95_seconds"]
            / columns["pim_joint_worst"]["ttft"]["p95_seconds"]),
    }
    pass_gate = all(value >= args.service_gain_floor
                    for value in speedup.values())
    total_decode_tokens = sum(stages.values())
    result = {
        "schema_version": "w12-request-bank-ledger-v1",
        "complete": True,
        "requests": len(requests),
        "service_gain_floor": args.service_gain_floor,
        "gate": {
            "passed": pass_gate,
            "rule": "target and joint-worst service/TTFT/P95 each beat the zero-cost Host control lower bound",
            "speedup": speedup,
        },
        "columns": columns,
        "decode_inventory": {
            "requests": len(requests),
            "decode_tokens": total_decode_tokens,
            "tokens_by_stage": dict(stages),
            "matrices_per_token": matrix_count,
            "weight_rows_per_token": rows_per_token,
            "host_bf16_weight_bytes_per_token": host_bf16_weight_bytes_per_token,
            "w12_logical_weight_bytes_per_token": w12_logical_weight_bytes_per_token,
            "internal_bank_weight_bytes_per_token": internal_bank_bytes_per_token,
            "external_activation_output_bytes_per_token": external_bytes_per_token,
            "refresh_cycles_per_token": int(totals["refresh_cycles"]),
            "layer_fence_cycles_per_token": int(totals["layer_fence_cycles"]),
            "bf16_reconstruction_cycles_per_token": int(
                totals.get("bf16_reconstruction_cycles", 0)),
            "bf16_rounding_cycles_per_token": int(
                totals.get("bf16_rounding_cycles", 0)),
            "bf16_group_reduction_cycles_per_token": int(
                totals.get("bf16_group_reduction_cycles", 0)),
            "bf16_mac_cycles_per_token": int(
                totals.get("bf16_mac_cycles", 0)),
            "bf16_pipeline_overlap_cycles_per_token": int(
                totals.get("bf16_pipeline_overlap_cycles", 0)),
            "refresh_ms_per_token": refresh_ms_per_token,
            "fence_ms_per_token": fence_ms_per_token,
            "dispatch_ms_per_token": dispatch_ms_per_token,
            "pim_linear_ms_per_token": float(
                target_arm["pim_linear_ms_per_token"]),
            "bf16_reconstruct_weights": bool(
                target_arm.get("bf16_reconstruct_weights", False)),
            "pipeline_bf16_reconstruction_and_mac": bool(
                target_arm.get("pipeline_bf16_reconstruction_and_mac", False)),
            "bf16_macs_per_bank_cycle": int(
                target_arm.get("bf16_macs_per_bank_cycle", 0)),
            "total_host_bf16_weight_stream_bytes": (
                total_decode_tokens * host_bf16_weight_bytes_per_token),
            "total_w12_logical_weight_stream_bytes": (
                total_decode_tokens * w12_logical_weight_bytes_per_token),
            "total_pim_external_bytes": (
                total_decode_tokens * external_bytes_per_token),
            "external_reduction_vs_host_bf16": (
                host_bf16_weight_bytes_per_token / external_bytes_per_token),
            "external_reduction_vs_w12_logical": (
                w12_logical_weight_bytes_per_token / external_bytes_per_token),
        },
        "request_ledger": requests,
        "boundary": [
            "request timing is a target-architecture projection on a measured 200-request trace",
            "Host-only and Host-control columns are measured/counterfactual Host baselines",
            "PIM columns include refresh, fences, dispatch and the declared joint-worst sensitivity",
            "internal bank-weight bytes are not external SoC traffic",
            "no physical PIM, RTL, phone, energy or thermal measurement was executed",
        ],
        "inputs_sha256": {
            str(args.projection): digest(args.projection),
            str(args.timeline): digest(args.timeline),
            **({str(args.raw_root): digest(args.raw_root)}
               if args.raw_root is not None and args.raw_root.is_file() else {}),
            **({str(path): digest(path)
                for path in sorted(args.raw_root.glob("user_*.json"))
                if not path.name.endswith(".partial.json")}
               if args.raw_root is not None and args.raw_root.is_dir() else {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "requests": len(requests),
        "gate_passed": pass_gate,
        "speedup": speedup,
        "external_reduction_vs_host_bf16": (
            host_bf16_weight_bytes_per_token / external_bytes_per_token),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
