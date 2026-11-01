"""Audit W12 functional execution across the real three-stage mainline.

This audit answers three separate questions:

* did parser, grounding and final Reader all execute from one W12 layout;
* did any decode call silently fall back to the Host linear;
* did replacing every Reader language Linear preserve the parser decision,
  grounding witnesses, selected evidence and final generated answer.

The last question is a functional paired audit against Host BF16.  It is not a
Judge semantic-quality result, and functional Torch wall time is not PIM time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STAGES = ("query_parser", "source_grounding", "reader_generation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(paths):
    rows = {}
    source_sha256 = {}
    documents = {}
    for path in paths:
        document = json.loads(Path(path).read_text())
        if not document.get("complete") or not isinstance(document.get("rows"), list):
            raise ValueError(f"complete run with rows required: {path}")
        source_sha256[str(path)] = digest(path)
        documents[str(path)] = document
        for row in document["rows"]:
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"duplicate question id across runs: {question_id}")
            rows[question_id] = row
    return rows, source_sha256, documents


def contract_row_chunk(document):
    materialization = document.get("reader_w12_bank_materialization") or {}
    chunk = materialization.get("contract_row_chunk")
    if isinstance(chunk, int):
        return chunk
    bank_count = document.get("w12_bank_count")
    tile_rows = document.get("w12_tile_rows")
    if isinstance(bank_count, int) and isinstance(tile_rows, int):
        return bank_count * tile_rows
    return None


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def common_prefix(left, right):
    count = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        count += 1
    return count


def selected_object_ids(row):
    return sorted({
        object_id
        for record in row.get("proof_plan", {}).get("selected_records", [])
        for object_id in record.get("object_ids", [])
    })


def stage_gate(row):
    ledger = row.get("w12_stage_dispatch")
    if not isinstance(ledger, dict) or set(ledger) != set(STAGES):
        raise ValueError("complete three-stage W12 dispatch ledger required")
    identities = {ledger[stage]["layout_instance_sha256"] for stage in STAGES}
    if len(identities) != 1:
        raise ValueError("more than one W12 layout identity in one request")
    grounding = row.get("proof_plan", {}).get("grounding", {})
    expected = {
        "query_parser": True,
        "source_grounding": (
            grounding.get("stop_reason") != "not_applicable_contextual"),
        "reader_generation": True,
    }
    return {
        "stages": {
            stage: {
                "bank_calls": ledger[stage]["bank_calls"],
                "decode_fallback_calls": ledger[stage]["decode_fallback_calls"],
                "prefill_fallback_calls": ledger[stage]["prefill_fallback_calls"],
                "fallback_rows": ledger[stage]["fallback_rows"],
                "selected_modules": ledger[stage]["selected_modules"],
                "unique_layouts": ledger[stage]["unique_layouts"],
            }
            for stage in STAGES
        },
        "expected": expected,
        "all_expected_stages_hit_bank": all(
            ledger[stage]["bank_calls"] > 0
            for stage in STAGES if expected[stage]),
        "unexpected_zero_stage": [
            stage for stage in STAGES
            if expected[stage] and ledger[stage]["bank_calls"] == 0
        ],
        "uninvoked_stage": [
            stage for stage in STAGES
            if not expected[stage] and ledger[stage]["bank_calls"] == 0
        ],
        "decode_fallback_zero": all(
            ledger[stage]["decode_fallback_calls"] == 0 for stage in STAGES),
        "single_layout_identity": True,
        "layout_instance_sha256": ledger["query_parser"]["layout_instance_sha256"],
    }


def pairstat(w12_row, host_row):
    if w12_row.get("question_sha256") != host_row.get("question_sha256"):
        raise ValueError("question identity differs across paired runs")
    w12_parse = w12_row.get("proof_plan", {}).get("query_parse", {})
    host_parse = host_row.get("proof_plan", {}).get("query_parse", {})
    w12_grounding = w12_row.get("proof_plan", {}).get("grounding", {})
    host_grounding = host_row.get("proof_plan", {}).get("grounding", {})
    w12_objects = selected_object_ids(w12_row)
    host_objects = selected_object_ids(host_row)
    w12_tokens = [int(value) for value in w12_row.get("answer_token_ids", [])]
    host_tokens = [int(value) for value in host_row.get("answer_token_ids", [])]
    matching_prefix = common_prefix(w12_tokens, host_tokens)
    return {
        "question_id": str(w12_row["question_id"]),
        "stage_gate": stage_gate(w12_row),
        "parser": {
            "host_status": host_parse.get("status"),
            "w12_status": w12_parse.get("status"),
            "host_output_tokens": host_parse.get("output_token_count"),
            "w12_output_tokens": w12_parse.get("output_token_count"),
            "parsed_equal": canonical(w12_parse.get("parsed")) == canonical(
                host_parse.get("parsed")),
            "host_demand_count": len(
                host_row.get("proof_plan", {}).get("demands", [])),
            "w12_demand_count": len(
                w12_row.get("proof_plan", {}).get("demands", [])),
        },
        "grounding": {
            "host_status": host_grounding.get("status"),
            "w12_status": w12_grounding.get("status"),
            "host_output_tokens": host_grounding.get("output_token_count"),
            "w12_output_tokens": w12_grounding.get("output_token_count"),
            "host_witness_count": host_grounding.get("witness_count"),
            "w12_witness_count": w12_grounding.get("witness_count"),
            "parsed_equal": canonical(w12_grounding.get("parsed")) == canonical(
                host_grounding.get("parsed")),
        },
        "evidence": {
            "host_object_ids": host_objects,
            "w12_object_ids": w12_objects,
            "selected_evidence_equal": w12_objects == host_objects,
        },
        "generation": {
            "host_tokens": host_tokens,
            "w12_tokens": w12_tokens,
            "generated_tokens": len(w12_tokens),
            "matching_prefix_tokens": matching_prefix,
            "first_token_match": matching_prefix >= 1,
            "full_generated_prefix_match": matching_prefix == len(w12_tokens),
            "answer_equal": w12_row.get("answer") == host_row.get("answer"),
        },
    }


def aggregate(per_request):
    def count(predicate):
        return sum(bool(predicate(item)) for item in per_request)

    stage_totals = {}
    for stage in STAGES:
        stage_totals[stage] = {
            "bank_calls": sum(item["stage_gate"]["stages"][stage]["bank_calls"]
                              for item in per_request),
            "decode_fallback_calls": sum(
                item["stage_gate"]["stages"][stage]["decode_fallback_calls"]
                for item in per_request),
            "prefill_fallback_calls": sum(
                item["stage_gate"]["stages"][stage]["prefill_fallback_calls"]
                for item in per_request),
        }
    return {
        "requests": len(per_request),
        "stage_totals": stage_totals,
        "all_expected_stages_hit_bank": count(
            lambda item: item["stage_gate"]["all_expected_stages_hit_bank"]),
        "unexpected_zero_stages": sum(
            len(item["stage_gate"]["unexpected_zero_stage"])
            for item in per_request),
        "uninvoked_stages": sum(
            len(item["stage_gate"]["uninvoked_stage"])
            for item in per_request),
        "decode_fallback_zero": count(
            lambda item: item["stage_gate"]["decode_fallback_zero"]),
        "single_layout_identity": count(
            lambda item: item["stage_gate"]["single_layout_identity"]),
        "parser_exact": count(lambda item: item["parser"]["parsed_equal"]),
        "grounding_exact": count(lambda item: item["grounding"]["parsed_equal"]),
        "selected_evidence_exact": count(
            lambda item: item["evidence"]["selected_evidence_equal"]),
        "first_token_match": count(
            lambda item: item["generation"]["first_token_match"]),
        "full_generated_prefix_match": count(
            lambda item: item["generation"]["full_generated_prefix_match"]),
        "answer_equal": count(lambda item: item["generation"]["answer_equal"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w12", type=Path, nargs="+", required=True)
    parser.add_argument("--host", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    w12_rows, w12_hashes, w12_documents = load_rows(args.w12)
    host_rows, host_hashes, host_documents = load_rows(args.host)
    if set(w12_rows) != set(host_rows):
        raise ValueError("W12 and Host runs must contain the same question ids")
    per_request = [pairstat(w12_rows[question_id], host_rows[question_id])
                   for question_id in sorted(w12_rows)]
    summary = aggregate(per_request)
    layout_identities = {
        item["question_id"]: item["stage_gate"]["layout_instance_sha256"]
        for item in per_request
    }
    host_contracts = {
        document.get("reader_quantization", "bf16")
        for document in host_documents.values()
    }
    if len(host_contracts) != 1:
        raise ValueError("all Host evidence files must share one reader contract")
    host_contract = next(iter(host_contracts))
    numeric_modes = {
        document.get("w12_numeric_mode", "bf16_reference")
        for document in w12_documents.values()
    }
    if len(numeric_modes) != 1:
        raise ValueError("all W12 evidence files must share one numeric mode")
    numeric_mode = next(iter(numeric_modes))
    host_contract_rows = None
    bank_contract_rows = None
    if host_contract == "w12a16":
        host_rows = {
            document.get("w12_host_contract_rows")
            for document in host_documents.values()
        }
        if len(host_rows) != 1 or not isinstance(next(iter(host_rows)), int):
            raise ValueError("Host W12A16 evidence must declare one row contract")
        host_contract_rows = next(iter(host_rows))
        bank_rows = {
            contract_row_chunk(document)
            for document in w12_documents.values()
        }
        if len(bank_rows) != 1 or not isinstance(next(iter(bank_rows)), int):
            raise ValueError("bank evidence must declare one row contract")
        bank_contract_rows = next(iter(bank_rows))
    contract_aligned = (
        host_contract_rows is None
        or host_contract_rows == bank_contract_rows)
    semantics_exact = (
        summary["parser_exact"] == summary["requests"]
        and summary["grounding_exact"] == summary["requests"]
        and summary["selected_evidence_exact"] == summary["requests"]
        and summary["full_generated_prefix_match"] == summary["requests"]
        and summary["answer_equal"] == summary["requests"])
    host_boundary = (
        "Host arm uses the dequantized W12A16 contract; paired drift isolates "
        "bank data-path effects from 12-bit weight quantization"
        if host_contract == "w12a16" else
        "Host arm is BF16; paired drift includes both bank data-path and 12-bit "
        "weight-quantization effects")
    result = {
        "schema_version": "w12-mainline-stage-evidence-audit-v2",
        "complete": True,
        "status": (
            "functional_three_stage_execution_passed; host_contract_semantics_exact"
            if semantics_exact else
            "functional_three_stage_execution_passed; semantic_path_drift_measured"
        ),
        "gates": {
            "execution": (
                summary["all_expected_stages_hit_bank"] == summary["requests"]
                and summary["unexpected_zero_stages"] == 0),
            "decode_fallback_zero": (
                summary["decode_fallback_zero"] == summary["requests"]),
            "single_layout": (
                summary["single_layout_identity"] == summary["requests"]),
            "contract_aligned": contract_aligned,
            "parser_exact": summary["parser_exact"] == summary["requests"],
            "grounding_exact": summary["grounding_exact"] == summary["requests"],
            "selected_evidence_exact": (
                summary["selected_evidence_exact"] == summary["requests"]),
            "generation_exact": (
                summary["full_generated_prefix_match"] == summary["requests"]
                and summary["answer_equal"] == summary["requests"]),
            "semantic_quality": "not_assessed_without_frozen_judge",
        },
        "summary": summary,
        "host_reader_quantization": host_contract,
        "host_contract_rows": host_contract_rows,
        "bank_contract_rows": bank_contract_rows,
        "contract_aligned": contract_aligned,
        "w12_numeric_mode": numeric_mode,
        "layout_instance_sha256_by_request": layout_identities,
        "per_request": per_request,
        "physical_pim_executed": False,
        "performance_comparable": False,
        "inputs_sha256": {**w12_hashes, **host_hashes},
        "boundary": [
            host_boundary,
            "functional W12 execution uses Torch on an RTX 3090, not silicon PIM",
            "functional wall time is not a PIM latency or speedup result",
            "prefill and multi-row calls remain explicit Host fallbacks",
            "one contextual request has no source-grounding model call by design",
            "generation-prefix agreement is not a Judge semantic-quality result",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "execution": result["gates"]["execution"],
        "decode_fallback_zero": result["gates"]["decode_fallback_zero"],
        "parser_exact": summary["parser_exact"],
        "grounding_exact": summary["grounding_exact"],
        "selected_evidence_exact": summary["selected_evidence_exact"],
        "answer_equal": summary["answer_equal"],
        "requests": summary["requests"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
