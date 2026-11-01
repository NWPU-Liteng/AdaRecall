"""Build the auditable entry point for the W12 bank closure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-evaluation", type=Path, required=True)
    parser.add_argument("--holdout-evaluation", type=Path, required=True)
    parser.add_argument("--functional-audit", type=Path, required=True)
    parser.add_argument("--request-ledger", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mainline-fp32-audit", type=Path, required=True)
    parser.add_argument("--mainline-bf16ref-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-failed", type=int, required=True)
    parser.add_argument("--full-tests-passed", type=int, required=True)
    parser.add_argument("--full-tests-failed", type=int, required=True)
    parser.add_argument("--known-unrelated-failure", required=True)
    args = parser.parse_args()

    dev = read_json(args.dev_evaluation)
    holdout = read_json(args.holdout_evaluation)
    functional = read_json(args.functional_audit)
    ledger = read_json(args.request_ledger)
    ablations = read_json(args.ablations)
    calibration = read_json(args.calibration)
    mainline_fp32 = read_json(args.mainline_fp32_audit)
    mainline_bf16ref = read_json(args.mainline_bf16ref_audit)
    if not functional.get("complete") or not ledger.get("complete"):
        raise ValueError("complete functional audit and request ledger required")
    if not ablations.get("complete") or not calibration.get("complete"):
        raise ValueError("complete ablation and calibration evidence required")
    if not mainline_fp32.get("complete") or not mainline_bf16ref.get("complete"):
        raise ValueError("complete mainline numeric-mode audits required")

    dev_quality = dev["quality"]
    holdout_quality = holdout["quality"]
    speedup = ledger["gate"]["speedup"]
    inventory = ledger["decode_inventory"]
    full_arm = ablations["arms"]["w12_full_decoder_shared"]
    reader_arm = ablations["arms"]["w12_reader_only_pim"]
    single_copy = ablations["capacity"]["copies"]["single_shared_decoder"]
    three_copies = ablations["capacity"]["copies"]["three_stage_copies"]
    numeric = functional["numeric_envelope"]
    prefix = functional["prefix_comparison"]

    mainline_dir = "artifacts/hyperpim_sc_w12_mainline_stage_smoke_20261029"
    mainline_questions = [
        "user_0", "0_q_184", "0_q_190", "0_q_230", "0_q_256", "0_q_258",
    ]

    root = Path(__file__).resolve().parents[1]
    files = [
        "README.md",
        "artifacts/hyperpim_sc_w12_dev192_20261029/evaluation/summary.json",
        "artifacts/hyperpim_sc_w12_holdout300_20261029/evaluation/summary.json",
        "artifacts/hyperpim_sc_w12_functional_20261029/w12_bank_functional_audit_v1.json",
        "artifacts/hyperpim_sc_w12_functional_20261029/w12_request_ledger_200_v2.json",
        "artifacts/hyperpim_sc_w12_functional_20261029/w12_system_ablations_v2.json",
        "artifacts/hyperpim_sc_w12_functional_20261029/w12_device_calibration_v3.json",
        "artifacts/hyperpim_sc_w12_bank_smoke_v2_20261029/baseline_user_0.json",
        "artifacts/hyperpim_sc_w12_shadow_smoke_20261029/baseline_user_0.json",
        "artifacts/hyperpim_sc_w12_bank_quality6_prefix8_20261029/baseline_user_0.json",
        "artifacts/hyperpim_sc_w12_bank_shadow6_prefix8_20261029/baseline_user_0.json",
        "pim_backend/w12_bank_layout.py",
        "pim_backend/w12_functional_dispatch.py",
        "scripts/run_omni_raw_image_smoke.py",
        "scripts/build_w12_request_ledger.py",
        "scripts/project_w12_system_ablations.py",
        "scripts/audit_w12_device_calibration.py",
        "scripts/audit_w12_bank_functional_evidence.py",
        "scripts/build_w12_bank_closure_manifest.py",
        "configs/edge_lpddr_pim_w12_target_v1.json",
        "tests/test_w12_bank_layout.py",
        "tests/test_w12_functional_dispatch.py",
        "tests/test_w12_system_evidence.py",
        "tests/test_w12_bank_functional_evidence.py",
        "tests/test_w12_mainline_stage_evidence.py",
        "scripts/run_proof_mainline.py",
        "scripts/audit_w12_mainline_stage_evidence.py",
        f"{mainline_dir}/w12_mainline_stage_audit_6q_v2.json",
        f"{mainline_dir}/w12_mainline_bf16ref_audit_6q_v1.json",
        "reports/HYPERPIM_SC_W12_BANK_CLOSURE_20261029_ZH.md",
        "reports/HYPERPIM_SC_W12_MAINLINE_STAGE_20261029_ZH.md",
        "reports/HYPERPIM_SC_W12_CLOSURE_RESULT_20261029_ZH.md",
        "reports/HYPERPIM_SC_W12_DECISION_20261029_ZH.md",
        "reports/HYPERPIM_SC_W12_FAILURE_DISCLOSURE_20261029_ZH.md",
        "reports/HYPERPIM_SC_W12_NEXT_CLOSURE_PLAN_20261029_ZH.md",
        "reports/HYPERPIM_SC_EVIDENCE_V3_20261029_ZH.md",
    ]
    for name in mainline_questions:
        w12_name = "user_0.json" if name == "user_0" else f"{name}_w12.json"
        files.extend([
            f"{mainline_dir}/{w12_name}",
            f"{mainline_dir}/{name}_host_bf16.json",
            f"{mainline_dir}/{name}_w12_bf16ref.json",
        ])
    file_records = {}
    for name in files:
        path = root / name
        if not path.is_file():
            raise ValueError(f"closure file missing: {name}")
        file_records[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}

    manifest = {
        "schema": "hyperpim-sc-w12-bank-closure-manifest-v2",
        "status": (
            "quality_and_functional_bank_closed; mainline_three_stage_execution_closed; "
            "fp32_semantic_drift_rejected; bf16_numeric_path_improved_not_closed; "
            "physical_pim_and_prefill_pending"
        ),
        "quality_qualified": bool(dev["gate"]["passed"] and holdout["gate"]["passed"]),
        "functional_bank_execution": True,
        "decode_bank_fallback_zero": bool(
            functional["gates"]["execution"]
            and functional["shadow_dispatch"]["decode_fallback_calls"] == 0
            and functional["replace_dispatch"]["decode_fallback_calls"] == 0
        ),
        "functional_numeric_envelope": bool(
            functional["gates"]["numeric_envelope"]),
        "mainline_three_stage_execution": bool(
            mainline_fp32["gates"]["execution"]
            and mainline_bf16ref["gates"]["execution"]),
        "mainline_decode_fallback_zero": bool(
            mainline_fp32["gates"]["decode_fallback_zero"]
            and mainline_bf16ref["gates"]["decode_fallback_zero"]),
        "mainline_fp32_semantic_drift": {
            "parser_exact": mainline_fp32["summary"]["parser_exact"],
            "grounding_exact": mainline_fp32["summary"]["grounding_exact"],
            "selected_evidence_exact": mainline_fp32["summary"][
                "selected_evidence_exact"],
            "answer_equal": mainline_fp32["summary"]["answer_equal"],
            "requests": mainline_fp32["summary"]["requests"],
        },
        "mainline_bf16ref_semantic_drift": {
            "parser_exact": mainline_bf16ref["summary"]["parser_exact"],
            "grounding_exact": mainline_bf16ref["summary"]["grounding_exact"],
            "selected_evidence_exact": mainline_bf16ref["summary"][
                "selected_evidence_exact"],
            "answer_equal": mainline_bf16ref["summary"]["answer_equal"],
            "requests": mainline_bf16ref["summary"]["requests"],
        },
        "recommended_bank_numeric_mode": "bf16_reference",
        "mainline_bank_replace_quality_qualified": False,
        "request_level_target_latency_gate": bool(ledger["gate"]["passed"]),
        "physical_pim_executed": False,
        "end_to_end_measured_latency_qualified": False,
        "prefill_pim_executed": False,
        "full_300_bank_replace_quality_qualified": False,
        "development": {
            "requests": dev["requests"],
            "reference_correct": dev_quality["reference_correct"],
            "candidate_correct": dev_quality["candidate_correct"],
            "paired_delta_pp": dev_quality["paired_delta_pp"],
            "ci95_pp": dev_quality["paired_delta_bootstrap_ci95_pp"],
            "gate_passed": bool(dev["gate"]["passed"]),
        },
        "holdout": {
            "requests": holdout["requests"],
            "reference_correct": holdout_quality["reference_correct"],
            "candidate_correct": holdout_quality["candidate_correct"],
            "paired_delta_pp": holdout_quality["paired_delta_pp"],
            "ci95_pp": holdout_quality["paired_delta_bootstrap_ci95_pp"],
            "gate_passed": bool(holdout["gate"]["passed"]),
        },
        "functional_bank": {
            "requests": prefix["requests"],
            "selected_modules": functional["shadow_dispatch"]["selected_modules"],
            "shadow_bank_calls": functional["shadow_dispatch"]["bank_calls"],
            "replace_bank_calls": functional["replace_dispatch"]["bank_calls"],
            "decode_fallback_calls": functional["replace_dispatch"]["decode_fallback_calls"],
            "prefill_fallback_calls": functional["replace_dispatch"]["prefill_fallback_calls"],
            "resident_bytes": functional["replace_dispatch"]["resident_bytes"],
            "released_source_parameter_bytes": (
                functional["replace_dispatch"]["released_source_parameter_bytes"]),
            "shadow_observations": numeric["observations"],
            "shadow_argmax_matches": numeric["argmax_matches"],
            "shadow_argmax_agreement": numeric["argmax_agreement"],
            "max_absolute_error": numeric["max_absolute_error"],
            "max_relative_l2_error": numeric["max_relative_l2_error"],
            "first_token_matches": prefix["first_token_matches"],
            "full_generated_prefix_matches": prefix["full_generated_prefix_matches"],
        },
        "request_projection": {
            "requests": inventory["requests"],
            "decode_tokens": inventory["decode_tokens"],
            "tokens_by_stage": inventory["tokens_by_stage"],
            "bf16_weight_bytes_per_token": inventory["host_bf16_weight_bytes_per_token"],
            "w12_logical_weight_bytes_per_token": (
                inventory["w12_logical_weight_bytes_per_token"]),
            "pim_external_bytes_per_token": (
                inventory["external_activation_output_bytes_per_token"]),
            "total_bf16_weight_stream_bytes": (
                inventory["total_host_bf16_weight_stream_bytes"]),
            "total_w12_logical_weight_stream_bytes": (
                inventory["total_w12_logical_weight_stream_bytes"]),
            "total_pim_external_bytes": inventory["total_pim_external_bytes"],
            "external_reduction_vs_host_bf16": (
                inventory["external_reduction_vs_host_bf16"]),
            "external_reduction_vs_w12_logical": (
                inventory["external_reduction_vs_w12_logical"]),
            "target_service_speedup": speedup["target_service_vs_host_control"],
            "target_ttft_speedup": speedup["target_ttft_vs_host_control"],
            "target_service_p95_speedup": (
                speedup["target_service_p95_vs_host_control"]),
            "target_ttft_p95_speedup": speedup["target_ttft_p95_vs_host_control"],
            "joint_worst_service_speedup": (
                speedup["joint_worst_service_vs_host_control"]),
            "joint_worst_ttft_speedup": (
                speedup["joint_worst_ttft_vs_host_control"]),
            "joint_worst_service_p95_speedup": (
                speedup["joint_worst_service_p95_vs_host_control"]),
            "joint_worst_ttft_p95_speedup": (
                speedup["joint_worst_ttft_p95_vs_host_control"]),
        },
        "system_ablations": {
            "full_decoder_shared_ttft": full_arm["speedup_vs_host_control"]["ttft_sum"],
            "reader_only_ttft": reader_arm["speedup_vs_host_control"]["ttft_sum"],
            "reader_only_gate_passed": bool(
                reader_arm["speedup_vs_host_control"]["ttft_sum"] >= 1.05),
            "single_copy_resident_bytes": single_copy["resident_bytes"],
            "single_copy_headroom_bytes": single_copy["headroom_bytes"],
            "single_copy_feasible": bool(single_copy["feasible"]),
            "three_stage_copies_resident_bytes": three_copies["resident_bytes"],
            "three_stage_copies_headroom_bytes": three_copies["headroom_bytes"],
            "three_stage_copies_feasible": bool(three_copies["feasible"]),
        },
        "device_calibration": {
            "calibration_status": calibration["calibration_status"],
            "physical_device_validated": bool(
                calibration["physical_device_validated"]),
            "full_pool_fill_seconds": calibration["cold_load"]["full_pool_fill_seconds"],
            "weight_plane_only_seconds": (
                calibration["cold_load"]["weight_plane_only_seconds"]),
            "refresh_fraction_of_pim_linear": (
                calibration["timing"]["refresh_fraction_of_pim_linear"]),
        },
        "tests": {
            "suite": "W12 closure regression",
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "full_suite": {
                "passed": args.full_tests_passed,
                "failed": args.full_tests_failed,
                "known_unrelated_failure": args.known_unrelated_failure,
            },
        },
        "files": file_records,
        "boundary": [
            "quality is conditional on the observed development and sealed users",
            "the 300-request arm is a Host W12 contract replay, not a bank replace rerun",
            "functional bank execution uses Torch on an RTX 3090; it is not physical PIM",
            "the mainline three-stage sample is six development questions with eight generated tokens",
            "fp32_bank changes parser, grounding and evidence decisions relative to Host BF16",
            "bf16_reference improves parser parity but does not close grounding or answer drift",
            "all request-level TTFT/service/P95 numbers are target-model projections",
            "prefill, vision and multi-row calls remain Host fallbacks",
            "the FP32 bank arithmetic is not bitwise equivalent to the BF16 dequantized Host contract",
            "the target projection does not yet charge a bank-local BF16 reconstruction datapath",
            "no RTL, Ramulator, phone, energy or thermal measurement was executed",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "closure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = []
    for name in sorted(file_records):
        lines.append(f"{file_records[name]['sha256']}  {name}")
    lines.append(f"{digest(manifest_path)}  artifacts/hyperpim_sc_w12_closure_20261029/closure_manifest.json")
    (args.output_dir / "sha256sums.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "quality_qualified": manifest["quality_qualified"],
        "functional_bank_execution": manifest["functional_bank_execution"],
        "mainline_three_stage_execution": manifest["mainline_three_stage_execution"],
        "mainline_bf16ref_parser_exact": (
            manifest["mainline_bf16ref_semantic_drift"]["parser_exact"]),
        "request_level_target_latency_gate": manifest["request_level_target_latency_gate"],
        "physical_pim_executed": manifest["physical_pim_executed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
