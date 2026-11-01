#!/usr/bin/env python3
"""Assemble the audited W12 tile-aligned closure manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def file_record(path):
    path = Path(path)
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
    }


def audit_gate(audit, name):
    gates = audit.get("gates", {})
    required = (
        "execution",
        "decode_fallback_zero",
        "single_layout",
        "contract_aligned",
        "parser_exact",
        "grounding_exact",
        "selected_evidence_exact",
        "generation_exact",
    )
    if not audit.get("complete") or not all(gates.get(key) for key in required):
        raise ValueError(f"mainline audit does not pass its exactness gates: {name}")
    summary = audit["summary"]
    exact_counts = (
        summary["parser_exact"],
        summary["grounding_exact"],
        summary["selected_evidence_exact"],
        summary["full_generated_prefix_match"],
        summary["answer_equal"],
        summary["decode_fallback_zero"],
        summary["single_layout_identity"],
    )
    if any(value != summary["requests"] for value in exact_counts):
        raise ValueError(f"mainline audit summary is inconsistent: {name}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-diagnostic", type=Path, required=True)
    parser.add_argument("--mainline-audit", type=Path, nargs="+", required=True)
    parser.add_argument("--request-ledger", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dev-quality", type=Path, required=True)
    parser.add_argument("--holdout-quality", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, nargs="*", default=[])
    parser.add_argument("--evidence", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    args = parser.parse_args()

    matrix = read_json(args.matrix_diagnostic)
    ledger = read_json(args.request_ledger)
    ablations = read_json(args.ablations)
    calibration = read_json(args.calibration)
    dev_quality = read_json(args.dev_quality)
    holdout_quality = read_json(args.holdout_quality)
    tests = read_json(args.tests)
    audits = [read_json(path) for path in args.mainline_audit]

    matrix_summary = matrix.get("summary", {})
    matrix_exact = bool(
        matrix.get("complete")
        and matrix_summary.get("matrices_added") == 249
        and matrix_summary.get("decode_bitwise_equal")
        == matrix_summary.get("decode_checks")
        and matrix_summary.get("prefill_fallback_bitwise_equal")
        == matrix_summary.get("prefill_fallback_checks")
        and matrix_summary.get("all_checks_bitwise_equal"))
    if not matrix_exact:
        raise ValueError("249-matrix decoder contract is not bitwise exact")

    mainline = [audit_gate(audit, str(path))
                for audit, path in zip(audits, args.mainline_audit)]
    if not ledger.get("complete") or not ledger["gate"]["passed"]:
        raise ValueError("request-level W12 ledger does not pass")
    if not ablations.get("complete") or not ablations["gate"]["passed"]:
        raise ValueError("W12 system ablations do not pass")
    single_copy = ablations["capacity"]["copies"]["single_shared_decoder"]
    if not single_copy["feasible"]:
        raise ValueError("single shared decoder does not fit the target pool")
    if calibration.get("physical_device_validated"):
        raise ValueError("manifest generator expects the current non-silicon boundary")
    for name, quality in (("dev", dev_quality), ("holdout", holdout_quality)):
        if not quality.get("complete") or not quality["quality"]["noninferiority_passed"]:
            raise ValueError(f"{name} quality noninferiority gate did not pass")
    if tests.get("effective_failed") != 0:
        raise ValueError("closure tests must pass apart from the declared unrelated failure")

    evidence = [
        args.matrix_diagnostic,
        *args.mainline_audit,
        args.request_ledger,
        args.ablations,
        args.calibration,
        args.dev_quality,
        args.holdout_quality,
        args.tests,
        *args.evidence,
    ]
    for root in args.evidence_root:
        evidence.extend(
            path for path in sorted(Path(root).rglob("*.json"))
            if not path.name.endswith(".partial.json"))
    unique = sorted({Path(path).resolve() for path in evidence})
    files = {relative(path): file_record(path) for path in unique}

    speedup = ledger["gate"]["speedup"]
    inventory = ledger["decode_inventory"]
    manifest = {
        "schema_version": "w12-tilealigned-closure-v1",
        "complete": True,
        "claim_level": "functional_exactness_plus_target_model_end_to_end",
        "gates": {
            "decoder_matrix_contract_exact": matrix_exact,
            "functional_mainline_exact": True,
            "single_shared_decoder_capacity": True,
            "request_level_target_gate": True,
            "quality_noninferiority": True,
            "tests_passed": True,
            "physical_pim_executed": False,
        },
        "decoder_contract": {
            "matrices": matrix_summary["matrices_added"],
            "decode_checks": matrix_summary["decode_checks"],
            "decode_bitwise_equal": matrix_summary["decode_bitwise_equal"],
            "prefill_fallback_checks": matrix_summary["prefill_fallback_checks"],
            "prefill_fallback_bitwise_equal": matrix_summary[
                "prefill_fallback_bitwise_equal"],
            "max_absolute_error": matrix_summary["max_absolute_error"],
            "contract": matrix["contract"],
        },
        "mainline_exactness": {
            "audits": len(mainline),
            "requests": sum(item["requests"] for item in mainline),
            "parser_exact": sum(item["parser_exact"] for item in mainline),
            "grounding_exact": sum(item["grounding_exact"] for item in mainline),
            "selected_evidence_exact": sum(
                item["selected_evidence_exact"] for item in mainline),
            "generated_prefix_exact": sum(
                item["full_generated_prefix_match"] for item in mainline),
            "answer_equal": sum(item["answer_equal"] for item in mainline),
            "decode_fallback_zero": sum(
                item["decode_fallback_zero"] for item in mainline),
        },
        "request_level_target": {
            "requests": ledger["requests"],
            "decode_tokens": inventory["decode_tokens"],
            "matrices_per_token": inventory["matrices_per_token"],
            "host_bf16_weight_bytes_per_token": inventory[
                "host_bf16_weight_bytes_per_token"],
            "pim_external_bytes_per_token": inventory[
                "external_activation_output_bytes_per_token"],
            "external_reduction_vs_host_bf16": inventory[
                "external_reduction_vs_host_bf16"],
            "bf16_reconstruction_cycles_per_token": inventory[
                "bf16_reconstruction_cycles_per_token"],
            "bf16_rounding_cycles_per_token": inventory[
                "bf16_rounding_cycles_per_token"],
            "bf16_group_reduction_cycles_per_token": inventory[
                "bf16_group_reduction_cycles_per_token"],
            "bf16_mac_cycles_per_token": inventory["bf16_mac_cycles_per_token"],
            "bf16_pipeline_overlap_cycles_per_token": inventory[
                "bf16_pipeline_overlap_cycles_per_token"],
            "target_service_speedup_vs_host_control": speedup[
                "target_service_vs_host_control"],
            "target_ttft_speedup_vs_host_control": speedup[
                "target_ttft_vs_host_control"],
            "target_service_p95_speedup_vs_host_control": speedup[
                "target_service_p95_vs_host_control"],
            "target_ttft_p95_speedup_vs_host_control": speedup[
                "target_ttft_p95_vs_host_control"],
            "joint_worst_service_speedup_vs_host_control": speedup[
                "joint_worst_service_vs_host_control"],
            "joint_worst_ttft_speedup_vs_host_control": speedup[
                "joint_worst_ttft_vs_host_control"],
            "joint_worst_service_p95_speedup_vs_host_control": speedup[
                "joint_worst_service_p95_vs_host_control"],
            "joint_worst_ttft_p95_speedup_vs_host_control": speedup[
                "joint_worst_ttft_p95_vs_host_control"],
        },
        "capacity": {
            "single_shared_decoder_bytes": single_copy["resident_bytes"],
            "single_shared_decoder_headroom_bytes": single_copy["headroom_bytes"],
            "three_stage_copies_feasible": ablations["capacity"]["copies"][
                "three_stage_copies"]["feasible"],
        },
        "quality": {
            "dev": dev_quality["quality"],
            "holdout": holdout_quality["quality"],
        },
        "device": {
            "calibration_status": calibration["calibration_status"],
            "physical_device_validated": calibration["physical_device_validated"],
            "cold_load": calibration["cold_load"],
            "timing": calibration["timing"],
        },
        "tests": tests,
        "files": files,
        "boundary": [
            "functional bank execution uses Torch on an RTX 3090, not silicon PIM",
            "request-level TTFT and service values are target-model projections anchored to a measured Host trace",
            "prefill and multi-row calls remain explicit Host fallbacks",
            "quality gates replay the Host W12A16 contract; they are not a new bank-replace Judge run",
            "no phone, RTL, Ramulator, energy or thermal measurement was executed",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "closure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    checksums = [f"{record['sha256']}  {name}" for name, record in sorted(files.items())]
    checksums.append(f"{digest(manifest_path)}  {relative(manifest_path)}")
    (args.output_dir / "sha256sums.txt").write_text("\n".join(checksums) + "\n")
    print(json.dumps({
        "output": relative(manifest_path),
        "requests": manifest["mainline_exactness"]["requests"],
        "decoder_matrices": manifest["decoder_contract"]["matrices"],
        "joint_worst_ttft": speedup["joint_worst_ttft_vs_host_control"],
        "files": len(files),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
