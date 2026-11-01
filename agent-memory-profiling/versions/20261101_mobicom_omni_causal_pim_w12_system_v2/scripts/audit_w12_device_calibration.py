#!/usr/bin/env python3
"""Audit W12 target-timeline parameters against the declared edge PIM profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.hyperpim_timeline import HyperPimTimelineConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--request-ledger", type=Path, required=True)
    parser.add_argument("--capacity-audit", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh calibration output required")

    timeline = json.loads(args.timeline.read_text())
    ledger = json.loads(args.request_ledger.read_text())
    capacity_audit = json.loads(args.capacity_audit.read_text())
    profile = json.loads(args.profile.read_text())
    config = profile["config"]
    base = HyperPimTimelineConfig()
    expected = {
        "stacks": base.stacks,
        "channels_per_stack": base.channels_per_stack,
        "banks_per_channel": base.banks_per_channel,
        "bank_count": base.bank_count,
        "clock_hz": base.clock_hz,
        "row_bytes": base.row_bytes,
        "t_rcd_cycles": base.t_rcd_cycles,
        "t_rp_cycles": base.t_rp_cycles,
        "t_rrd_cycles": base.t_rrd_cycles,
        "t_faw_cycles": base.t_faw_cycles,
        "refresh_interval_cycles": base.refresh_interval_cycles,
        "refresh_block_cycles": base.refresh_block_cycles,
        "bank_internal_bytes_per_cycle": base.bank_internal_bytes_per_cycle,
        "int8_macs_per_bank_cycle": base.int8_macs_per_bank_cycle,
        "conversion_lanes_per_bank": base.conversion_lanes_per_bank,
        "bf16_macs_per_bank_cycle": base.bf16_macs_per_bank_cycle,
        "bf16_reconstruction_lanes_per_bank": (
            base.bf16_reconstruction_lanes_per_bank),
        "bf16_round_lanes_per_bank": base.bf16_round_lanes_per_bank,
        "bf16_group_reduction_lanes_per_bank": (
            base.bf16_group_reduction_lanes_per_bank),
        "bf16_reconstruct_weights": True,
        "pipeline_bf16_reconstruction_and_mac": True,
        "matrix_command_cycles": base.matrix_command_cycles,
        "layer_fence_cycles": base.layer_fence_cycles,
        "segment_fences_per_layer": base.segment_fences_per_layer,
    }
    mismatches = {
        key: {"profile": config.get(key), "timeline": expected_value}
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError("target profile and timeline parameters differ: "
                         + ",".join(sorted(mismatches)))

    decoder = ledger["decode_inventory"]
    w12_plane_bytes = int(decoder["w12_logical_weight_bytes_per_token"])
    w12_pool_bytes = int(
        capacity_audit["variants"]["w12a16"][
            "pool_with_bf16_embedding_alias_bytes"])
    capacity_bytes = int(profile["physical_capacity_bytes"])
    capacity_ok = w12_pool_bytes <= capacity_bytes
    required_external = (float(profile["external_service_decimal_gb_s"]) * 1e9)
    pool_fill_seconds = w12_pool_bytes / required_external
    weight_plane_seconds = w12_plane_bytes / required_external
    refresh_ms = float(decoder["refresh_ms_per_token"])
    fence_ms = float(decoder["fence_ms_per_token"])
    pim_ms = float(timeline["arms"]["w12a16"]["pim_linear_ms_per_token"])
    joint_ms = float(
        timeline["arms"]["w12a16_joint_worst"]["pim_linear_ms_per_token"])
    speedup = ledger["gate"]["speedup"]
    joint_gate = bool(
        speedup["joint_worst_service_vs_host_control"] >= 1.05
        and speedup["joint_worst_ttft_vs_host_control"] >= 1.05)
    result = {
        "schema_version": "w12-device-calibration-audit-v1",
        "complete": True,
        "profile": profile["name"],
        "calibration_status": profile["calibration_status"],
        "physical_device_validated": False,
        "parameter_match": {
            "passed": True,
            "expected": expected,
            "mismatches": mismatches,
        },
        "capacity": {
            "w12_weight_plane_bytes": w12_plane_bytes,
            "w12_pool_with_bf16_embedding_alias_bytes": w12_pool_bytes,
            "target_pool_bytes": capacity_bytes,
            "headroom_bytes": capacity_bytes - w12_pool_bytes,
            "passed": capacity_ok,
            "boundary": "logical W12 weight plane against the reserved pool; metadata and repair are separate",
        },
        "timing": {
            "clock_hz": base.clock_hz,
            "bank_count": base.bank_count,
            "t_rcd_cycles": base.t_rcd_cycles,
            "t_rp_cycles": base.t_rp_cycles,
            "t_rrd_cycles": base.t_rrd_cycles,
            "t_faw_cycles": base.t_faw_cycles,
            "refresh_interval_cycles": base.refresh_interval_cycles,
            "refresh_block_cycles": base.refresh_block_cycles,
            "refresh_ms_per_token": refresh_ms,
            "fence_ms_per_token": fence_ms,
            "refresh_fraction_of_pim_linear": refresh_ms / pim_ms,
            "pim_linear_ms_per_token": pim_ms,
            "joint_worst_pim_linear_ms_per_token": joint_ms,
            "joint_worst_gate_passed": joint_gate,
        },
        "cold_load": {
            "external_service_gbps": float(
                profile["external_service_decimal_gb_s"]),
            "weight_plane_only_seconds": weight_plane_seconds,
            "full_pool_fill_seconds": pool_fill_seconds,
            "amortization": (
                "charged once per cold start or model refresh, not once per request"
            ),
            "boundary": (
                "one serial external fill of the declared resident pool, including "
                "the BF16 embedding alias; no overlap, compression or request reuse "
                "multiplication"
            ),
        },
        "gate": {
            "passed": bool(capacity_ok and joint_gate),
            "rule": "capacity fits the declared pool and the joint-worst request projection still beats Host control by at least 1.05x in service and TTFT",
        },
        "boundary": profile["boundaries"],
        "inputs_sha256": {
            str(args.timeline): digest(args.timeline),
            str(args.request_ledger): digest(args.request_ledger),
            str(args.capacity_audit): digest(args.capacity_audit),
            str(args.profile): digest(args.profile),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "gate_passed": result["gate"]["passed"],
        "capacity_headroom_bytes": result["capacity"]["headroom_bytes"],
        "refresh_fraction_of_pim_linear": result["timing"]["refresh_fraction_of_pim_linear"],
        "full_pool_fill_seconds": pool_fill_seconds,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
