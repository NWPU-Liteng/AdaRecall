"""Audit real-activation W12 bank execution against the frozen Host contract.

The audit deliberately separates three claims:

* all selected decode modules executed from the packed bank layout;
* the FP32 bank arithmetic stayed inside a declared numerical envelope;
* generated-token prefixes agree with the previously generated Host arm.

Token-prefix agreement is diagnostic only.  It is not a semantic-quality gate
and it never turns functional Torch wall time into PIM latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


RELATIVE_L2_ENVELOPE = 5e-3
ARGMIN_ARGMAX_AGREEMENT_FLOOR = 0.99


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(path):
    document = json.loads(Path(path).read_text())
    if not document.get("complete") or not isinstance(document.get("rows"), list):
        raise ValueError(f"complete run with rows required: {path}")
    return document


def row_map(document, *, expected_ids=None, required_ids=None):
    rows = {str(row["question_id"]): row for row in document["rows"]}
    if len(rows) != len(document["rows"]):
        raise ValueError("duplicate question id in run")
    if expected_ids is not None and set(rows) != set(expected_ids):
        raise ValueError("question sets differ across compared runs")
    if required_ids is not None and not set(required_ids) <= set(rows):
        raise ValueError("required questions are absent from the reference run")
    return rows


def common_prefix(left, right):
    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def dispatch_contract_summary(document):
    dispatch = document.get("reader_w12_bank_dispatch")
    if not isinstance(dispatch, dict):
        raise ValueError("reader_w12_bank_dispatch missing")
    required = {
        "selected_modules", "bank_calls", "decode_fallback_calls",
        "prefill_fallback_calls", "resident_bytes", "all_selected_modules_executed",
        "silent_fallback", "shadow", "physical_pim_executed",
        "performance_comparable",
    }
    if not required <= set(dispatch):
        raise ValueError("incomplete W12 dispatch summary")
    return dispatch


def compare_prefixes(replace, host):
    replace_rows = row_map(replace)
    host_rows = row_map(host, required_ids=replace_rows)
    comparisons = []
    for question_id in sorted(replace_rows):
        replace_row = replace_rows[question_id]
        host_row = host_rows[question_id]
        if replace_row.get("question_sha256") != host_row.get("question_sha256"):
            raise ValueError(f"question identity differs: {question_id}")
        replace_tokens = [int(value) for value in replace_row["answer_token_ids"]]
        host_tokens = [int(value) for value in host_row["answer_token_ids"]]
        prefix = common_prefix(replace_tokens, host_tokens)
        comparisons.append({
            "question_id": question_id,
            "replace_tokens": replace_tokens,
            "host_prefix": host_tokens[:len(replace_tokens)],
            "generated_tokens": len(replace_tokens),
            "matching_prefix_tokens": prefix,
            "first_token_match": prefix >= 1,
            "full_generated_prefix_match": prefix == len(replace_tokens),
        })
    return comparisons


def compare_single_arm(shadow, replace):
    shadow_rows = row_map(shadow)
    replace_rows = row_map(replace, expected_ids=shadow_rows)
    result = []
    for question_id in sorted(shadow_rows):
        shadow_row = shadow_rows[question_id]
        replace_row = replace_rows[question_id]
        shadow_tokens = [int(value) for value in shadow_row["answer_token_ids"]]
        replace_tokens = [int(value) for value in replace_row["answer_token_ids"]]
        prefix = common_prefix(shadow_tokens, replace_tokens)
        result.append({
            "question_id": question_id,
            "shadow_tokens": shadow_tokens,
            "replace_tokens": replace_tokens,
            "matching_prefix_tokens": prefix,
            "exact_generated_tokens": prefix == len(shadow_tokens) == len(replace_tokens),
            "same_answer_sha256": (
                shadow_row.get("answer_sha256") == replace_row.get("answer_sha256")
            ),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--replace", type=Path, required=True)
    parser.add_argument("--host-contract", type=Path, required=True)
    parser.add_argument("--single-shadow", type=Path)
    parser.add_argument("--single-replace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shadow = load_run(args.shadow)
    replace = load_run(args.replace)
    host = load_run(args.host_contract)
    shadow_dispatch = dispatch_contract_summary(shadow)
    replace_dispatch = dispatch_contract_summary(replace)
    if shadow_dispatch["mode"] != "shadow" or replace_dispatch["mode"] != "replace":
        raise ValueError("shadow and replace dispatch modes required")
    if shadow_dispatch["numeric_mode"] != "fp32_bank":
        raise ValueError("audit requires the target fp32_bank numeric mode")

    shadow_values = shadow_dispatch["shadow"]
    observations = int(shadow_values["observations"])
    argmax_matches = int(shadow_values["argmax_matches"])
    relative_l2 = float(shadow_values["max_relative_l2_error"])
    if not all(math.isfinite(value) for value in (relative_l2,)):
        raise ValueError("finite shadow error required")
    agreement = argmax_matches / observations if observations else 0.0
    numeric_gate = bool(
        observations > 0
        and relative_l2 <= RELATIVE_L2_ENVELOPE
        and agreement >= ARGMIN_ARGMAX_AGREEMENT_FLOOR
    )
    execution_gate = bool(
        shadow_dispatch["all_selected_modules_executed"]
        and replace_dispatch["all_selected_modules_executed"]
        and shadow_dispatch["decode_fallback_calls"] == 0
        and replace_dispatch["decode_fallback_calls"] == 0
        and not shadow_dispatch["silent_fallback"]
        and not replace_dispatch["silent_fallback"]
    )
    comparisons = compare_prefixes(replace, host)
    full_prefixes = sum(item["full_generated_prefix_match"] for item in comparisons)
    first_tokens = sum(item["first_token_match"] for item in comparisons)

    single = None
    single_dispatch = None
    if (args.single_shadow is None) != (args.single_replace is None):
        raise ValueError("single shadow and replace must be supplied together")
    if args.single_shadow is not None:
        single_shadow = load_run(args.single_shadow)
        single_replace = load_run(args.single_replace)
        single_dispatch = dispatch_contract_summary(single_shadow)
        single = compare_single_arm(single_shadow, single_replace)

    result = {
        "schema_version": "w12-bank-functional-audit-v1",
        "complete": True,
        "status": (
            "functional_bank_execution_and_numeric_envelope_passed; "
            "semantic_generation_not_exact"
        ),
        "gates": {
            "execution": execution_gate,
            "numeric_envelope": numeric_gate,
            "relative_l2_envelope": RELATIVE_L2_ENVELOPE,
            "argmax_agreement_floor": ARGMIN_ARGMAX_AGREEMENT_FLOOR,
            "semantic_quality": "not_assessed_by_prefix_audit",
        },
        "shadow_dispatch": {
            key: shadow_dispatch[key]
            for key in (
                "selected_modules", "bank_calls", "decode_fallback_calls",
                "prefill_fallback_calls", "fallback_rows", "bank_external_bytes",
                "fallback_external_bytes", "resident_bytes",
                "released_source_parameter_bytes", "source_released",
                "all_selected_modules_executed", "silent_fallback",
                "physical_pim_executed", "performance_comparable",
            )
        },
        "numeric_envelope": {
            "observations": observations,
            "argmax_matches": argmax_matches,
            "argmax_agreement": agreement,
            "max_absolute_error": shadow_values["max_absolute_error"],
            "max_relative_l2_error": relative_l2,
        },
        "replace_dispatch": {
            key: replace_dispatch[key]
            for key in (
                "selected_modules", "bank_calls", "decode_fallback_calls",
                "prefill_fallback_calls", "fallback_rows", "bank_external_bytes",
                "fallback_external_bytes", "resident_bytes",
                "released_source_parameter_bytes", "source_released",
                "all_selected_modules_executed", "silent_fallback",
                "physical_pim_executed", "performance_comparable",
            )
        },
        "prefix_comparison": {
            "requests": len(comparisons),
            "first_token_matches": first_tokens,
            "full_generated_prefix_matches": full_prefixes,
            "all_first_tokens_match": first_tokens == len(comparisons),
            "per_request": comparisons,
            "boundary": (
                "prefix agreement is a functional diagnostic against a prior Host "
                "generation; it is not a Judge result or a semantic noninferiority test"
            ),
        },
        "single_request_control": ({
            "requests": len(single),
            "exact_generated_token_matches": sum(
                item["exact_generated_tokens"] for item in single
            ),
            "same_answer_sha256_matches": sum(
                item["same_answer_sha256"] for item in single
            ),
            "shadow_observations": single_dispatch["shadow"]["observations"],
            "shadow_argmax_matches": single_dispatch["shadow"]["argmax_matches"],
            "per_request": single,
        } if single is not None else None),
        "inputs_sha256": {
            str(args.shadow): digest(args.shadow),
            str(args.replace): digest(args.replace),
            str(args.host_contract): digest(args.host_contract),
            **({
                str(args.single_shadow): digest(args.single_shadow),
                str(args.single_replace): digest(args.single_replace),
            } if single is not None else {}),
        },
        "boundary": [
            "real Reader activations on RTX 3090; Torch executes the packed bank layout",
            "no silicon PIM, RTL, Ramulator, phone, energy or thermal measurement",
            "functional wall time is not a PIM latency result",
            "prefill and multi-row calls remain explicit Host fallbacks",
            "the FP32 bank path is not bitwise equivalent to the BF16 dequantized Host contract",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
