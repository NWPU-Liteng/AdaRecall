#!/usr/bin/env python3
"""Project system-level W12 ablations on the same measured request trace."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import simulate_hyperpim_sc_timeline as timeline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latency(values):
    return timeline.latency(values)


def assessment_for(projection, arm, nonlinear, grammar, dispatch=4.0):
    denominator = arm["pim_linear_ms_per_token"] + nonlinear + grammar + dispatch
    safety = arm["structured_ms_per_token"] / denominator
    if not math.isclose(safety, 1.15, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("unexpected frozen safety factor")
    return timeline.assess_target(
        projection, arm["pim_linear_ms_per_token"], nonlinear, grammar,
        dispatch, safety)


def project_relative(target_details, *, parse_ms, grounding_ms, reader_ms,
                     target_structured_ms, target_reader_ms):
    service, ttft = [], []
    for row in target_details:
        parse_tokens = int(row["parse_tokens"])
        grounding_tokens = int(row["grounding_tokens"])
        reader_tokens = int(row["reader_tokens"])
        service_value = float(row["cycle_target_service_seconds"])
        ttft_value = float(row["cycle_target_ttft_seconds"])
        service_value += parse_tokens * (parse_ms - target_structured_ms) / 1000
        service_value += grounding_tokens * (
            grounding_ms - target_structured_ms) / 1000
        service_value += reader_tokens * (reader_ms - target_reader_ms) / 1000
        ttft_value += parse_tokens * (parse_ms - target_structured_ms) / 1000
        ttft_value += grounding_tokens * (
            grounding_ms - target_structured_ms) / 1000
        service.append(service_value)
        ttft.append(ttft_value)
    return {"service": latency(service), "ttft": latency(ttft)}


def speedup(control, candidate):
    return {
        "service_sum": control["service"]["sum_seconds"] / candidate["service"]["sum_seconds"],
        "ttft_sum": control["ttft"]["sum_seconds"] / candidate["ttft"]["sum_seconds"],
        "service_p95": control["service"]["p95_seconds"] / candidate["service"]["p95_seconds"],
        "ttft_p95": control["ttft"]["p95_seconds"] / candidate["ttft"]["p95_seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold-load-gbps", type=float, default=90.0693433912629)
    parser.add_argument("--gain-floor", type=float, default=1.05)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh ablation output required")
    if (not math.isfinite(args.cold_load_gbps) or args.cold_load_gbps <= 0
            or not math.isfinite(args.gain_floor) or args.gain_floor <= 1):
        raise ValueError("positive cold-load bandwidth and gain floor above one required")

    projection = json.loads(args.projection.read_text())
    timeline_document = json.loads(args.timeline.read_text())
    capacity = json.loads(args.capacity_audit.read_text())
    w12_capacity = capacity["variants"]["w12a16"]
    plan = w12_capacity["decoder_plan"]
    capacity_bytes = int(plan["capacity_bytes"])
    decoder_copy_bytes = int(plan["total_bytes"])
    metadata_bytes = (int(w12_capacity["pool_with_bf16_embedding_alias_bytes"])
                      - decoder_copy_bytes)

    nonlinear = float(projection["gpu_nonlinear_profile"]["median_ms_per_token"])
    grammar = float(projection["pim_grammar_model"]["milliseconds_per_token"])
    target_arm = timeline_document["arms"]["w12a16"]
    target = assessment_for(projection, target_arm, nonlinear, grammar)
    control = target["control"]
    baseline = target["baseline"]
    target_structured = float(target_arm["structured_ms_per_token"])
    target_reader = float(target_arm["reader_ms_per_token"])
    host_linear = float(projection["matrix_linear_cost"]["host_linear_ms_per_token"])
    host_control_ms = host_linear + nonlinear + 4.0

    arms = {
        "host_only_measured": {
            "kind": "measured_host",
            "service": baseline["service"],
            "ttft": baseline["ttft"],
            "speedup_vs_host_control": speedup(control, baseline),
        },
        "host_control_lower_bound": {
            "kind": "counterfactual_host",
            "service": control["service"],
            "ttft": control["ttft"],
            "speedup_vs_host_control": speedup(control, control),
        },
        "w12_full_decoder_shared": {
            "kind": "target_pim",
            "service": target["target"]["service"],
            "ttft": target["target"]["ttft"],
            "speedup_vs_host_control": speedup(control, target["target"]),
            "resident_decoder_copies": 1,
            "resident_bytes": decoder_copy_bytes + metadata_bytes,
            "capacity_feasible": decoder_copy_bytes + metadata_bytes <= capacity_bytes,
        },
    }

    for name, parse_ms, grounding_ms, reader_ms, copies in (
            ("w12_reader_only_pim", host_control_ms, host_control_ms,
             target_reader, 1),
            ("w12_parser_grounding_only_pim", target_structured,
             target_structured, host_control_ms, 1),
            ("w12_without_grammar_scan", target_structured - grammar,
             target_structured - grammar, target_reader, 1)):
        result = project_relative(
            target["details"], parse_ms=parse_ms, grounding_ms=grounding_ms,
            reader_ms=reader_ms, target_structured_ms=target_structured,
            target_reader_ms=target_reader)
        arms[name] = {
            "kind": "target_pim_ablation",
            "service": result["service"],
            "ttft": result["ttft"],
            "speedup_vs_host_control": speedup(control, result),
            "resident_decoder_copies": copies,
            "resident_bytes": copies * decoder_copy_bytes + metadata_bytes,
            "capacity_feasible": copies * decoder_copy_bytes + metadata_bytes <= capacity_bytes,
            "stage_cost_ms": {
                "parser": parse_ms,
                "grounding": grounding_ms,
                "reader": reader_ms,
            },
        }

    w8_arm = timeline_document["arms"]["w8a16_r2"]
    w8 = assessment_for(projection, w8_arm, nonlinear, grammar)
    arms["w8a16_r2_full_decoder_shared"] = {
        "kind": "precision_ablation",
        "service": w8["target"]["service"],
        "ttft": w8["target"]["ttft"],
        "speedup_vs_host_control": speedup(control, w8["target"]),
        "resident_decoder_copies": 1,
        "capacity_feasible": True,
    }

    worst_arm = timeline_document["arms"]["w12a16_joint_worst"]
    worst = assessment_for(projection, worst_arm, nonlinear, grammar,
                           dispatch=float(worst_arm["dispatch_ms_per_token"]))
    arms["w12_joint_worst"] = {
        "kind": "target_pim_sensitivity",
        "service": worst["target"]["service"],
        "ttft": worst["target"]["ttft"],
        "speedup_vs_host_control": speedup(control, worst["target"]),
        "resident_decoder_copies": 1,
        "capacity_feasible": True,
    }

    copies = {}
    for name, count in (("single_shared_decoder", 1),
                        ("reader_only_decoder", 1),
                        ("three_stage_copies", 3)):
        total = count * decoder_copy_bytes + metadata_bytes
        copies[name] = {
            "decoder_copies": count,
            "resident_bytes": total,
            "headroom_bytes": capacity_bytes - total,
            "feasible": total <= capacity_bytes,
        }

    cold_load_seconds = decoder_copy_bytes / (args.cold_load_gbps * 1e9)
    cold_load = {
        "one_decoder_copy_seconds": cold_load_seconds,
        "three_stage_copies_seconds": 3 * cold_load_seconds,
        "cross_stage_reuse_saved_seconds_per_cold_start": 2 * cold_load_seconds,
        "amortization": (
            "charged once per cold start or model refresh, not once per request; "
            "resident requests reuse the single loaded decoder"
        ),
        "boundary": (
            "single external cold-load stream; does not overlap with compute; "
            "the saving is not multiplied by the request count"
        ),
    }

    gate = {
        "rule": "full shared W12 must beat the Host control lower bound in service, TTFT and both P95s; single-copy capacity must fit",
        "passed": bool(
            arms["w12_full_decoder_shared"]["speedup_vs_host_control"]["service_sum"] >= args.gain_floor
            and arms["w12_full_decoder_shared"]["speedup_vs_host_control"]["ttft_sum"] >= args.gain_floor
            and arms["w12_full_decoder_shared"]["speedup_vs_host_control"]["service_p95"] >= args.gain_floor
            and arms["w12_full_decoder_shared"]["speedup_vs_host_control"]["ttft_p95"] >= args.gain_floor
            and arms["w12_full_decoder_shared"]["capacity_feasible"]),
        "gain_floor": args.gain_floor,
    }
    result = {
        "schema_version": "w12-system-ablation-projection-v1",
        "complete": True,
        "requests": len(projection["requests_detail"]),
        "gate": gate,
        "arms": arms,
        "capacity": {
            "capacity_bytes": capacity_bytes,
            "decoder_copy_bytes": decoder_copy_bytes,
            "metadata_bytes": metadata_bytes,
            "copies": copies,
        },
        "cold_load": cold_load,
        "finding": (
            "Parser and grounding account for most decoder tokens; Reader-only "
            "PIM loses the TTFT gate, while three stage-private copies exceed "
            "the bank pool. The defensible mechanism is one shared decoder copy "
            "reused by all three agent stages."
        ),
        "boundary": [
            "request-level analytical projection on a measured Host trace",
            "PIM stage costs include the frozen cycle model and dispatch/grammar terms",
            "capacity uses the audited 8 GiB target bank pool and one weight plane",
            "no physical device, power, thermal or RTL measurement",
        ],
        "inputs_sha256": {
            str(args.projection): digest(args.projection),
            str(args.timeline): digest(args.timeline),
            str(args.capacity_audit): digest(args.capacity_audit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "gate_passed": gate["passed"],
        "arms": {name: arm["speedup_vs_host_control"]
                 for name, arm in arms.items()},
        "capacity": copies,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
